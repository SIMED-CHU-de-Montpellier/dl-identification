import random
import torch.distributed as dist
from torch.utils.data import Sampler


class PKBatchSampler(Sampler):
    """
    Produces indices for batches of size P * K:
      - Randomly sample P distinct labels
      - For each label, sample K indices from that class (with cycling)
    """
    def __init__(self, label_to_indices, P=16, K=4, drop_last=True, shuffle_labels=True):
        self.label_to_indices = {l: list(idxs) for l, idxs in label_to_indices.items() if len(idxs) >= 2}
        self.labels = list(self.label_to_indices.keys())
        if len(self.labels) < P:
            raise ValueError(f"Not enough identities ({len(self.labels)}) to sample P={P} per batch.")
        self.P = P
        self.K = K
        self.drop_last = drop_last
        self.shuffle_labels = shuffle_labels

        # Pre-create per-label cyclical pools so we can draw K images repeatedly
        self._label_pointers = {l: 0 for l in self.labels}
        self._label_pools = {}
        for l, idxs in self.label_to_indices.items():
            pool = idxs.copy()
            random.shuffle(pool)
            self._label_pools[l] = pool

    def __iter__(self):
        labels = self.labels.copy()
        if self.shuffle_labels:
            random.shuffle(labels)

        # Estimate how many batches we can produce this epoch
        # We’ll just produce as many as we can, sampling labels with replacement per batch.
        size = 0

        while size < len(self):
            if len(labels) < self.P and self.drop_last:
                return

            chosen_labels = random.sample(self.labels, self.P)
            batch_indices = []
            # For each chosen label, take K items (cycling through pool)
            for l in chosen_labels:
                pool = self._label_pools[l]
                ptr = self._label_pointers[l]
                take = []
                for _ in range(self.K):
                    take.append(pool[ptr])
                    ptr += 1
                    if ptr >= len(pool):
                        # cycle and reshuffle to keep diversity
                        random.shuffle(pool)
                        ptr = 0
                self._label_pointers[l] = ptr
                batch_indices.extend(take)

            size += 1
            yield batch_indices

    def __len__(self):
        # Unknown true length as we sample with replacement.
        # You can define an epoch length pragmatically:
        # e.g., total_images // (P*K)
        total_images = sum(len(v) for v in self.label_to_indices.values())
        return total_images // (self.P * self.K)



class DistributedPKBatchSampler(Sampler):
    """
    Produces indices for batches of size P * K:
      - Randomly sample P distinct labels
      - For each label, sample K indices from that class (with cycling)
      - Supports multi-GPU training via rank/world_size partitioning
    """
    def __init__(self, label_to_indices, P=16, K=4, drop_last=True, shuffle_labels=True):
        self.label_to_indices = {l: list(idxs) for l, idxs in label_to_indices.items() if len(idxs) >= 2}
        self.labels = list(self.label_to_indices.keys())
        if len(self.labels) < P:
            raise ValueError(f"Not enough identities ({len(self.labels)}) to sample P={P} per batch.")
        self.P = P
        self.K = K
        self.drop_last = drop_last
        self.shuffle_labels = shuffle_labels

        # Multi-GPU support
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Per-label cyclical pools
        self._label_pointers = {l: 0 for l in self.labels}
        self._label_pools = {}
        for l, idxs in self.label_to_indices.items():
            pool = idxs.copy()
            random.shuffle(pool)
            self._label_pools[l] = pool

    def __iter__(self):
        if self.shuffle_labels:
            random.shuffle(self.labels)

        # Generate all batches for all GPUs
        total_batches = len(self) * self.world_size
        all_batches = []

        while len(all_batches) < total_batches:
            chosen_labels = random.sample(self.labels, self.P)
            batch_indices = []
            for l in chosen_labels:
                pool = self._label_pools[l]
                ptr = self._label_pointers[l]
                for _ in range(self.K):
                    batch_indices.append(pool[ptr])
                    ptr += 1
                    if ptr >= len(pool):
                        random.shuffle(pool)
                        ptr = 0
                self._label_pointers[l] = ptr
            all_batches.append(batch_indices)

        # Each GPU gets its own slice
        my_batches = all_batches[self.rank::self.world_size]
        for batch in my_batches:
            yield batch

    def __len__(self):
        total_images = sum(len(v) for v in self.label_to_indices.values())
        return (total_images // (self.P * self.K)) // self.world_size