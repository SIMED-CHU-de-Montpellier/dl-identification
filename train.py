import pytorch_lightning as pl
import mlflow
import shutil
import os
import torch
import yaml

from data_module import ImageDataModule
from model import ImageRecognitionNet
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# PARAMETERS
# data
train_root_dir = cfg["data"]["train_path"]
valid_root_dir = cfg["data"]["val_path"]

# model
backbone = cfg["model"]["backbone"]
embedding_size = cfg["model"]["embedding_size"]
margin = cfg["model"]["margin"]
num_val_triplets = cfg["training"]["num_val_triplets"]
num_workers = cfg["training"]["num_workers"]
epochs = cfg["training"]["epochs"]
P = cfg["sampler"]["P"]
K = cfg["sampler"]["K"]
val_batch_size = cfg["validation"]["batch_size"]

# optimizer
optimizer = cfg["optimizer"]["name"]
backbone_lr = cfg["optimizer"]["backbone_lr"]
fc_lr = cfg["optimizer"]["fc_lr"]
wd = cfg["optimizer"]["weight_decay"]

# mlflow
experiment_name = f"test-{backbone}"
tracking_uri = cfg["mlflow"]["tracking_uri"]

# trainer
n_runs = cfg["runtime"]["n_runs"]
tmp_dir = cfg["runtime"]["tmp_dir"]


mlflow.set_tracking_uri(tracking_uri)
for run_idx in range(n_runs):
    print(f"\n{'=' * 50}")
    print(f"Training run {run_idx + 1}/{n_runs}")
    print(f"{'=' * 50}")

    # generate run_id
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        run_number = 1
    else:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        run_number = len(runs) + 1

    # data module
    dm = ImageDataModule(train_root_dir=train_root_dir, valid_root_dir=valid_root_dir, num_workers=num_workers,
                         num_valid_triplets=num_val_triplets, P=P, K=K, val_batch_size=val_batch_size)

    # model
    model = ImageRecognitionNet(
        margin=margin,
        backbone_lr=backbone_lr,
        fc_lr=fc_lr,
        epochs=epochs,
        embedding_size=embedding_size,
        backbone=backbone,
        optimizer=optimizer,
        wd=wd,
    )

    # mlflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"{backbone}-{run_number:03d}",
        tracking_uri=tracking_uri,
        log_model=False # I do it manually
    )

    # callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{tmp_dir}/checkpoints",
        filename=f"{backbone}"+"-{epoch:03d}-{val-acc:.4f}",
        monitor="val-acc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=2,
        callbacks=[ckpt_cb, lr_cb],
        logger=mlf_logger,
        default_root_dir=tmp_dir,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        use_distributed_sampler=False,
        precision="16-mixed", # AMP
    )
    trainer.fit(model, dm)

    # synchronize, wait for all gpus to end
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # upload to MLflow only on rank 0
    if trainer.is_global_zero:
        best_ckpt = ckpt_cb.best_model_path
        if best_ckpt:
            print(f"Uploading checkpoint to MLflow: {best_ckpt}")
            mlflow.log_artifact(
                best_ckpt,
                artifact_path="checkpoints",
                run_id=mlf_logger.run_id
            )

        # cleanup only on rank 0
        print(f"Cleaning up {tmp_dir} ...")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)