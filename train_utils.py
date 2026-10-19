import torch
import re
import os
import pandas as pd
from typing import List, Tuple


def read_folder(data_path, extensions = ('.jpg', '.jpeg', '.png')):
    rows = []
    for cls_name in sorted(os.listdir(data_path)):
        cls_dir = os.path.join(data_path, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        for f in os.listdir(cls_dir):
            if f.lower().endswith(extensions):
                img, ext = os.path.splitext(f)
                rows.append((cls_name, img, ext))
    df = pd.DataFrame(rows, columns=["name", "id", "ext"])
    df = df.sort_values(by=['name', 'id']).reset_index(drop=True)
    df['class'] = pd.factorize(df['name'])[0]
    return df


def get_path(root, item):
    return os.path.join(root, item['name'], item['id'] + item['ext'])


def pairwise_distances(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """
    Compute pairwise L2 distances.
    embeddings: (N, D) assumed L2-normalized
    returns: (N, N)
    """
    dists = torch.cdist(embeddings, embeddings, p=2)
    if squared:
        dists = dists.pow(2)
    return dists


def semihard_triplet_mining(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> Tuple[List[int], List[int], List[int], float]:
    """
    Miner:
      For each (anchor, positive) pair in the batch:
        choose a semi-hard negative: d_ap < d_an < d_ap + margin (pick the closest).
      If <4 triplets are found, increase the margin recursively by +0.1.
    Returns lists of indices (anchor_idx, pos_idx, neg_idx) and (maybe) increased margin.
    """

    device = embeddings.device
    labels = labels.view(-1)
    N = labels.size(0)
    dmat = pairwise_distances(embeddings, squared=False)  # (N, N)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    diff = ~same
    anchor_indices, pos_indices, neg_indices = [], [], []

    for i in range(N):
        pos_in_class = torch.where(same[i])[0]  # includes i
        pos_in_class = pos_in_class[pos_in_class != i]  # exclude self

        for j in pos_in_class.tolist():
            d_ap = dmat[i, j].item()
            neg_cands = torch.where(diff[i])[0].to(device)
            if neg_cands.numel() == 0:
                continue
            d_an = dmat[i, neg_cands]

            # semi-hard
            semi_mask = (d_an > d_ap) & (d_an < d_ap + margin)
            semi_cands = neg_cands[semi_mask]

            if semi_cands.numel() > 0:
                best_idx = torch.argmin(d_an[semi_mask]).item()
                n = semi_cands[best_idx].item()
            else:
                continue  # no fallback

            anchor_indices.append(i)
            pos_indices.append(j)
            neg_indices.append(n)

    if len(anchor_indices) <= 3:
        # Ensure at least a few triplets by expanding margin
        return semihard_triplet_mining(embeddings, labels, margin + 0.1)

    return anchor_indices, pos_indices, neg_indices, margin


# from huggingface trainer
# https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py
def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
    )
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result