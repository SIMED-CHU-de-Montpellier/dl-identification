import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import numpy as np
import re

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from typing import List, Tuple, Any
from train_utils import semihard_triplet_mining, get_parameter_names


class ImageRecognitionNet(pl.LightningModule):
    """
    Lightning version of FaceNetModel:
    - Backbone: ViT-L/16 (default) with pretrained weights (optional).
    - Head: Linear -> L2-normalized embedding.
    - Train: batch-hard triplet mined from PK batches (images, labels).
    - Val: static triplets dict (anc/pos/neg) with TripletMarginLoss.
    """

    def __init__(
        self,
        backbone: str = "r50",
        embedding_size: int = 256,
        margin: float = 0.2,
        backbone_lr: float = 1e-4,
        fc_lr: float = 1e-3,
        epochs = 20,
        optimizer = "adam",
        wd = 1e-2
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        match backbone:
            case "r50":
                self.backbone = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
                self.backbone.fc = nn.Identity()
                self.fc = nn.Linear(2048, embedding_size)
            case "r152":
                self.backbone = torchvision.models.resnet152(weights=torchvision.models.resnet.ResNet152_Weights.DEFAULT)
                self.backbone.fc = nn.Identity()
                self.fc = nn.Linear(2048, embedding_size)
            case "vitb16":
                self.backbone = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
                self.backbone.heads = nn.Identity()
                self.fc = nn.Linear(768, embedding_size)
            case "vitl16":
                self.backbone = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.DEFAULT)
                self.backbone.heads = nn.Identity()
                self.fc = nn.Linear(1024, embedding_size)
            case "maxvit":
                self.backbone = torchvision.models.maxvit_t(weights=torchvision.models.MaxVit_T_Weights.DEFAULT)
                self.backbone.classifier[5] = nn.Identity()
                self.fc = nn.Linear(512, embedding_size)
            case "swins":
                self.backbone = torchvision.models.swin_s(weights=torchvision.models.Swin_S_Weights.DEFAULT)
                self.backbone.head = nn.Identity()
                self.fc = nn.Linear(768, embedding_size)
            case "swinb":
                self.backbone = torchvision.models.swin_b(weights=torchvision.models.Swin_B_Weights.DEFAULT)
                self.backbone.head = nn.Identity()
                self.fc = nn.Linear(1024, embedding_size)
            case "convnexts":
                self.backbone = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.DEFAULT)
                self.backbone.classifier[-1] = nn.Identity()
                self.fc = nn.Linear(768, embedding_size)
            case "convnextb":
                self.backbone = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.DEFAULT)
                self.backbone.classifier[-1] = nn.Identity()
                self.fc = nn.Linear(1024, embedding_size)
            case "convnextl":
                self.backbone = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT)
                self.backbone.classifier[-1] = nn.Identity()
                self.fc = nn.Linear(1536, embedding_size)
            case "regnety8gf":
                self.backbone = torchvision.models.regnet_y_8gf(weights=torchvision.models.RegNet_Y_8GF_Weights.DEFAULT)
                self.backbone.fc = nn.Identity()
                self.fc = nn.Linear(2016, embedding_size)
            case "regnety32gf":
                self.backbone = torchvision.models.regnet_y_32gf(weights=torchvision.models.RegNet_Y_32GF_Weights.DEFAULT)
                self.backbone.fc = nn.Identity()
                self.fc = nn.Linear(3712, embedding_size)
            case _:
                print('Unknown backbone, defaulting to resnet50')
                self.backbone = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
                self.backbone.fc = nn.Identity()
                self.fc = nn.Linear(2048, embedding_size)

        # loss
        self.triplet = nn.TripletMarginLoss(margin=self.hparams.margin, p=2)
        self.l2 = nn.PairwiseDistance(2)

        # buffers
        self.train_pos_dists: List[float] = []
        self.train_neg_dists: List[float] = []
        self.val_pos_dists: List[float] = []
        self.val_neg_dists: List[float] = []
        self.train_triplet_count = 0
        self.val_triplet_count = 0


    def _stash_dists(self, phase: str, pos_d: torch.Tensor, neg_d: torch.Tensor):
        pos = pos_d.detach().float().view(-1).tolist()
        neg = neg_d.detach().float().view(-1).tolist()

        if phase == "train":
            self.train_pos_dists += pos
            self.train_neg_dists += neg
        else:
            self.val_pos_dists += pos
            self.val_neg_dists += neg


    def _compute_accuracy_from_buffers(self, phase: str) -> Tuple[float, float]:
        if phase == "train":
            pos = self.train_pos_dists
            neg = self.train_neg_dists
        else:
            pos = self.val_pos_dists
            neg = self.val_neg_dists

        if len(pos) == 0 or len(neg) == 0:
            return 0.0, 0.0

        dists = np.array(pos + neg, dtype=np.float32)
        labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)

        # Smaller distance => positive (same identity)
        uniq = np.unique(dists)
        thresholds = np.concatenate(([uniq[0] - 1e-6], uniq, [uniq[-1] + 1e-6]))

        best_acc, best_thr = 0.0, thresholds[0]
        for thr in thresholds:
            preds = (dists < thr).astype(np.int32)
            acc = (preds == labels).mean()
            if acc > best_acc:
                best_acc, best_thr = float(acc), float(thr)

        return best_acc, best_thr


    # -------------------
    # Forward = embeddings (L2-normalized)
    # -------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)        # [B, feat_dim]
        x = self.fc(x)              # [B, embedding_size]
        x = F.normalize(x, p=2, dim=1)
        return x


    def training_step(self, batch: Any, batch_idx: int):
        """
        Expect PK batches: (images, labels)
        1) embed (no grad) -> miner -> choose anc/pos/neg
        2) embed selected triplets (with grad) -> TripletMarginLoss
        """
        images, labels = batch  # from PK sampler

        # 1) miner on frozen graph
        self.eval()
        with torch.no_grad():
            embs = self(images)  # [B, D]
            a_idx, p_idx, n_idx, used_margin = semihard_triplet_mining(
                embs, labels, margin=self.hparams.margin
            )
        self.train()

            # anc_img = images[a_idx]
            # pos_img = images[p_idx]
            # neg_img = images[n_idx]

        # compute loss with grad
        # a, p, n = self(anc_img), self(pos_img), self(neg_img)

        # insane forward vram optimization
        a_idx_t = torch.tensor(a_idx, device=self.device)
        p_idx_t = torch.tensor(p_idx, device=self.device)
        n_idx_t = torch.tensor(n_idx, device=self.device)

        triplet_idx = torch.unique(torch.cat([a_idx_t, p_idx_t, n_idx_t]))
        triplet_imgs = images[triplet_idx]
        triplet_embs = self(triplet_imgs)

        idx_map = torch.zeros(images.size(0), dtype=torch.long, device=self.device)
        idx_map[triplet_idx] = torch.arange(len(triplet_idx), device=self.device)

        a = triplet_embs[idx_map[a_idx_t]]
        p = triplet_embs[idx_map[p_idx_t]]
        n = triplet_embs[idx_map[n_idx_t]]
        # end of optimization

        # Use possibly increased margin from the miner
        loss_fn = nn.TripletMarginLoss(margin=used_margin, p=2)
        loss = loss_fn(a, p, n)

        pos_d = self.l2(a, p)
        neg_d = self.l2(a, n)

        self._stash_dists("train", pos_d, neg_d)
        self.log("train-loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Important: weight epoch average by number of mined triplets
        self.train_triplet_count += len(a_idx)

        return loss


    def validation_step(self, batch: Any, batch_idx: int):
        anc, pos, neg = batch["anc_img"], batch["pos_img"], batch["neg_img"]
        a, p, n = self(anc), self(pos), self(neg)
        loss = self.triplet(a, p, n)
        pos_d = self.l2(a, p)
        neg_d = self.l2(a, n)
        self._stash_dists("val", pos_d, neg_d)

        # Weight by count of triplets in this batch
        self.val_triplet_count += anc.size(0)
        self.log("val-loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {"loss": loss}


    def on_train_epoch_start(self) -> None:
        self.train_pos_dists = []
        self.train_neg_dists = []
        self.train_triplet_count = 0


    def on_validation_epoch_start(self) -> None:
        self.val_pos_dists = []
        self.val_neg_dists = []
        self.val_triplet_count = 0


    def on_train_epoch_end(self) -> None:
        if self.train_triplet_count == 0:
            return
        acc, thr = self._compute_accuracy_from_buffers(phase="train")
        self.log("train-acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


    def on_validation_epoch_end(self) -> None:
        if self.val_triplet_count == 0:
            return
        acc, thr = self._compute_accuracy_from_buffers(phase="val")
        self.log("val-acc", acc, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)


    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam([
                {"params": self.backbone.parameters(), "lr": self.hparams.backbone_lr, "name": "backbone"},
                {"params": self.fc.parameters(), "lr": self.hparams.fc_lr, "name": "fc"},
            ])
        elif self.hparams.optimizer == "adamw":
            # avoid weight decay on norm and bias as per huggingface trainer
            # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
            forbidden_name_patterns = [
                r"bias",
                r"layernorm",
                r"rmsnorm",
                r"(?:^|\.)norm(?:$|\.)",
                r"_norm(?:$|\.)"
            ]

            # https://github.com/pytorch/vision/blob/main/references/classification/utils.py
            forbidden_layer_types = [
                nn.modules.batchnorm._BatchNorm,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.modules.instancenorm._InstanceNorm,
                nn.LocalResponseNorm,
            ]
            backbone_decay_parameters = get_parameter_names(self.backbone, forbidden_layer_types, forbidden_name_patterns)
            fc_decay_parameters = get_parameter_names(self.fc, forbidden_layer_types, forbidden_name_patterns)

            # don't decay bias and norm
            optimizer = torch.optim.AdamW([
                {
                    "params": [
                        p for n, p in self.backbone.named_parameters() if (n in backbone_decay_parameters and p.requires_grad)
                    ],
                    "lr": self.hparams.backbone_lr, "weight_decay": self.hparams.wd, "name": "backbone"
                },
                {
                    "params": [
                        p for n, p in self.backbone.named_parameters() if (n not in backbone_decay_parameters and p.requires_grad)
                    ],
                    "lr": self.hparams.backbone_lr, "weight_decay": 0.0, "name": "backbone_nd"
                },
                {
                    "params": [
                        p for n, p in self.fc.named_parameters() if (n in fc_decay_parameters and p.requires_grad)
                    ],
                    "lr": self.hparams.fc_lr, "weight_decay": 1e-2, "name": "fc"
                },
                {
                    "params": [
                        p for n, p in self.fc.named_parameters() if (n not in fc_decay_parameters and p.requires_grad)
                    ],
                    "lr": self.hparams.fc_lr, "weight_decay": 0.0, "name": "fc_nd"
                }
            ])

            # decay all
            # optimizer = torch.optim.AdamW([
            #     {"params": self.backbone.parameters(), "lr": self.hparams.backbone_lr, "weight_decay": self.hparams.wd, "name": "backbone"},
            #     {"params": self.fc.parameters(), "lr": self.hparams.fc_lr, "weight_decay": 1e-2, "name": "fc"},
            # ])

        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=2)
        cosine = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=0)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


    def _get_decay_nodecay_params(self, layer, patterns):
        decay, no_decay = [], []
        for name, param in layer.named_parameters():
            if not param.requires_grad:
                continue
            if any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay