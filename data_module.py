import pytorch_lightning as pl

from torch.utils.data import DataLoader, DistributedSampler
from dataset import ImageFolderDataset, ImageTripletDataset
from sampler import DistributedPKBatchSampler
from transforms import get_normalize_transform


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, train_root_dir, valid_root_dir, num_valid_triplets,
                 num_workers=4, P=16, K=4, val_batch_size=32):
        super().__init__()
        self.train_root = train_root_dir
        self.valid_root = valid_root_dir
        self.num_valid_triplets = num_valid_triplets

        self.num_workers = num_workers
        self.P = P
        self.K = K

        self.transform = get_normalize_transform()
        self.val_batch_size = val_batch_size


    def setup(self, stage=None):
        self.train_dataset = ImageFolderDataset(
            root_dir=self.train_root,
            transform=self.transform,
            min_images_per_class=2
        )
        self.train_sampler = DistributedPKBatchSampler(
            self.train_dataset.label_to_indices,
            P=self.P, K=self.K, drop_last=True, shuffle_labels=True
        )

        self.valid_dataset = ImageTripletDataset(
            root_dir=self.valid_root,
            num_triplets=self.num_valid_triplets,
            transform=self.transform
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )


    def val_dataloader(self):
        sampler = DistributedSampler(
            self.valid_dataset,
            shuffle=False,
        )

        return DataLoader(
            self.valid_dataset,
            batch_size=self.val_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )