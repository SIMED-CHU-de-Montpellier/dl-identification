import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from eval_utils import load_model, compute_embeddings, get_checkpoint_path
from transforms import get_normalize_transform

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# parameters
EXPERIMENT_NAME = cfg["validation"]["experiment-name"]
MLFLOW_TRACKING_URI = cfg["mlflow"]["tracking_uri"]
TEST_DATA_PATH = cfg["data"]["test_path"]
ARTIFACTS_BASE_PATH = cfg["mlflow"]["artifacts_base_path"]
batch_size = cfg["validation"]["batch_size"]
n_splits = cfg["validation"]["n_splits"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(embeddings, class_to_ids):
    db_labels, db_vecs, db_ids = [], [], []
    q_labels, q_vecs, q_ids = [], [], []

    for label, embs in embeddings.items():
        ids = class_to_ids[label]
        for i in range(len(embs) - 1):
            db_vecs.append(embs[i].numpy())
            db_labels.append(label)
            db_ids.append(ids[i])

        q_vecs.append(embs[-1].numpy())
        q_labels.append(label)
        q_ids.append(ids[-1])

    db_vecs = np.stack(db_vecs)
    q_vecs = np.stack(q_vecs)
    dists = np.linalg.norm(q_vecs[:, None, :] - db_vecs[None, :, :], axis=-1)

    rank1, rank5, rank10 = 0, 0, 0
    for i, q_label in enumerate(q_labels):
        sorted_idx = np.argsort(dists[i])
        top10_labels = [db_labels[j] for j in sorted_idx[:10]]

        if top10_labels[0] == q_label:
            rank1 += 1
        if q_label in top10_labels[:5]:
            rank5 += 1
        if q_label in top10_labels:
            rank10 += 1

    n = len(q_labels)

    return rank1 / n, rank5 / n, rank10 / n


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# experiment_names = ['panoramique-r50', 'panoramique-r152', 'panoramique-regnety8gf', 'panoramique-regnety32gf', 'panoramique-convnexts', 'panoramique-convnextb', 'panoramique-convnextl', 'panoramique-vitb16', 'panoramique-vitl16', 'panoramique-maxvit', 'panoramique-swins', 'panoramique-swinb']
experiment_names = [EXPERIMENT_NAME]

all_summaries = []
for experiment_name in experiment_names:
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string="attributes.status = 'FINISHED'")
    trfrm = get_normalize_transform()

    # loop over runs
    all_rank1, all_rank5, all_rank10 = [], [], []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        print(f"\n{'=' * 50}")
        print(f"Run: {run_name}")

        checkpoint_path = get_checkpoint_path(client, experiment, run, ARTIFACTS_BASE_PATH)
        model = load_model(checkpoint_path, device)
        embeddings, class_to_name, class_to_ids = compute_embeddings(
            TEST_DATA_PATH, model, device, trfrm, batch_size, return_mapping=True
        )

        rank1, rank5, rank10 = evaluate(embeddings, class_to_ids)
        print(f"  Acc@1  : {rank1:.4f}")
        print(f"  Acc@5  : {rank5:.4f}")
        print(f"  Acc@10 : {rank10:.4f}")

        all_rank1.append(rank1)
        all_rank5.append(rank5)
        all_rank10.append(rank10)

    metrics = {
        "Acc@1": all_rank1,
        "Acc@5": all_rank5,
        "Acc@10": all_rank10,
    }

    model_name = experiment_name[12:]
    summary = {
        name: f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        for name, values in metrics.items()
    }

    # Add model name as the first column
    summary = {"Model": model_name, **summary}
    all_summaries.append(summary)

df_markdown = pd.DataFrame(all_summaries)
print(df_markdown.to_markdown(index=False))