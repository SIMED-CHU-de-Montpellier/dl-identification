import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import ImageRecognitionNet
from train_utils import read_folder, get_path


class FlatImageDataset(Dataset):
    def __init__(self, df, data_path, transform):
        self.rows      = [row for _, row in df.iterrows()]
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(get_path(self.data_path, row))
        return self.transform(img), row['class']


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(checkpoint_path: str, device: torch.device) -> ImageRecognitionNet:
    model = ImageRecognitionNet.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def compute_embeddings(data_path: str, model, device: torch.device,
                       transform, batch_size: int = 256, return_mapping: bool = False):
    df = read_folder(data_path)
    classes = df['class'].unique()
    class_to_name = {}
    class_to_ids = {}

    # build flat lists preserving order
    for c in classes:
        df_class = df[df['class'] == c]
        class_to_name[c] = df_class.iloc[0]['name']
        class_to_ids[c] = df_class['id'].tolist()

    dataset = FlatImageDataset(df, data_path, transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    all_embs, all_classes = [], []

    print('Embedding pictures ...')
    for tensors, batch_classes in loader:
        with torch.no_grad():
            all_embs.append(model(tensors.to(device)).cpu())
        all_classes.extend(batch_classes.tolist())
    all_embs = torch.cat(all_embs)

    # reconstruct per-class embeddings
    embeddings = {}
    for c in classes:
        mask          = [i for i, cls in enumerate(all_classes) if cls == c]
        embeddings[c] = all_embs[mask]

    if return_mapping:
        return embeddings, class_to_name, class_to_ids
    else:
        return embeddings


def compute_pos_dists(embeddings: dict) -> np.ndarray:
    pos_dists = []
    for label, embs in embeddings.items():
        D = torch.cdist(embs, embs)
        rows, cols = torch.triu_indices(len(embs), len(embs), offset=1)
        pos_dists.extend(D[rows, cols].cpu().tolist())
    return np.array(pos_dists)


def sample_neg_dists(embeddings: dict, num_samples: int) -> np.ndarray:
    labels = list(embeddings.keys())
    person_pairs = [(l1, l2) for i, l1 in enumerate(labels) for l2 in labels[i + 1:]]
    samples, neg_dists = [], []
    while len(samples) < num_samples:
        pair_idx = np.random.randint(len(person_pairs))
        l1, l2 = person_pairs[pair_idx]
        idx1 = np.random.randint(embeddings[l1].shape[0])
        idx2 = np.random.randint(embeddings[l2].shape[0])
        pair = (l1, idx1, l2, idx2)
        if pair not in samples:
            samples.append(pair)
            dist = torch.dist(embeddings[l1][idx1], embeddings[l2][idx2], p=2)
            neg_dists.append(dist.item())
    return np.array(neg_dists)


def get_checkpoint_path(client, experiment, run, artifacts_path):
    artifacts = client.list_artifacts(run.info.run_id)
    for a in artifacts:
        if a.path.endswith(".ckpt"):
            return f"{artifacts_path}/{experiment.experiment_id}/{run.info.run_id}/artifacts/{a.path}"

    # fallback
    for a in artifacts:
        sub = client.list_artifacts(run.info.run_id, a.path)
        for s in sub:
            if s.path.endswith(".ckpt"):
                return f"{artifacts_path}/{experiment.experiment_id}/{run.info.run_id}/artifacts/{s.path}"
    raise FileNotFoundError(f"No checkpoint found for run {run.info.run_id}")