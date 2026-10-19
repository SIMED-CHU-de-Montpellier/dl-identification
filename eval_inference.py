import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow import MlflowClient

from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold

from eval_utils import set_seeds, load_model, compute_embeddings, compute_pos_dists, sample_neg_dists, get_checkpoint_path
from transforms import get_normalize_transform

SEED = 67

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

EXPERIMENT_NAME = cfg["validation"]["experiment-name"]
MLFLOW_TRACKING_URI = cfg["mlflow"]["tracking_uri"]
VAL_DATA_PATH = cfg["data"]["val_path"]
TEST_DATA_PATH = cfg["data"]["test_path"]
ARTIFACTS_BASE_PATH = cfg["mlflow"]["artifacts_base_path"]
batch_size = cfg["validation"]["batch_size"]
n_splits = cfg["validation"]["n_splits"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_eer_threshold(fold_classes, embeddings):
    fold_embs = {c: embeddings[c] for c in fold_classes}
    pos_dists = compute_pos_dists(fold_embs)
    neg_dists = sample_neg_dists(fold_embs, num_samples=len(pos_dists))
    y_true = np.concatenate([np.ones_like(pos_dists), np.zeros_like(neg_dists)])
    y_scores = np.concatenate([-pos_dists, -neg_dists])
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer_thr = -thr[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    y_pred = np.concatenate([pos_dists < eer_thr, neg_dists < eer_thr])
    return eer_thr, eer, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)


def compute_eer(pos_dists, neg_dists):
    y_true = np.concatenate([np.ones_like(pos_dists), np.zeros_like(neg_dists)])
    y_scores = np.concatenate([-pos_dists, -neg_dists])
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer


def get_threshold(embeddings):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    classes_array = np.array(list(embeddings.keys()))
    fold_thrs, fold_accs, fold_f1s, fold_eers  = [], [], [], []
    for _, val_idx in kf.split(classes_array):
        thr, eer, acc, f1 = compute_eer_threshold(classes_array[val_idx].tolist(), embeddings)
        fold_thrs.append(thr)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_eers.append(eer)

    return float(np.mean(fold_thrs)), float(np.mean(fold_eers)), float(np.mean(fold_accs)), float(np.mean(fold_f1s))


def evaluate(embeddings, threshold):
    pos_dists = compute_pos_dists(embeddings)
    neg_dists = sample_neg_dists(embeddings, num_samples=len(pos_dists))
    y_true = np.concatenate([np.ones_like(pos_dists), np.zeros_like(neg_dists)])
    y_pred = np.concatenate([pos_dists < threshold, neg_dists < threshold])

    tar = np.mean(pos_dists < threshold)
    far = np.mean(neg_dists < threshold)

    y_scores = np.concatenate([-pos_dists, -neg_dists])
    auc = roc_auc_score(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    tar_at_far1 = np.interp(0.01, fpr, tpr)
    # valid = fpr <= 0.01
    # tar_at_far1 = np.max(tpr[valid]) if np.any(valid) else 0.0

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred), compute_eer(pos_dists, neg_dists), tar, far, auc, tar_at_far1

if __name__ == '__main__':
    # experiment_names = ["panoramique-convnexts-split_005"] + [f"panoramique-convnexts-split_{i:03d}" for i in range(10, 100, 10)]
    # experiment_names = ['panoramique-r50', 'panoramique-r152', 'panoramique-regnety8gf', 'panoramique-regnety32gf',
    #                     'panoramique-convnexts', 'panoramique-convnextb', 'panoramique-convnextl', 'panoramique-vitb16',
    #                     'panoramique-vitl16', 'panoramique-maxvit', 'panoramique-swins', 'panoramique-swinb']
    experiment_names = [EXPERIMENT_NAME]

    client = MlflowClient()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    all_summaries = []
    for experiment_name in experiment_names:
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                                  filter_string="attributes.status = 'FINISHED'")
        trfrm = get_normalize_transform()

        # loop over runs
        val_accs, val_f1s, val_eers, all_thrs = [], [], [], []
        test_accs, test_f1s, test_eers, test_tars, test_fars, test_aucs, test_tarfar1s = [], [], [], [], [], [], []

        for run in runs:
            # reset the seed to ensure each run samples the same data points
            set_seeds(SEED)

            run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
            print(f"\n{'=' * 50}")
            print(f"Run: {run_name}")

            checkpoint_path = get_checkpoint_path(client, experiment, run, ARTIFACTS_BASE_PATH)
            model = load_model(checkpoint_path, device)

            # threshold on val
            print("  Computing threshold ...")
            val_embeddings = compute_embeddings(VAL_DATA_PATH, model, device, trfrm, batch_size)
            threshold, eer, acc, f1 = get_threshold(val_embeddings)
            all_thrs.append(threshold)
            val_accs.append(acc)
            val_f1s.append(f1)
            val_eers.append(eer)
            print(f"  Threshold: {threshold:.4f}")

            # evaluate on test
            print("  Evaluating on test set ...")
            test_embeddings = compute_embeddings(TEST_DATA_PATH, model, device, trfrm, batch_size)
            acc, f1, eer, tar, far, auc, tarfar1 = evaluate(test_embeddings, threshold)
            test_accs.append(acc)
            test_f1s.append(f1)
            test_eers.append(eer)
            test_tars.append(tar)
            test_fars.append(far)
            test_aucs.append(auc)
            test_tarfar1s.append(tarfar1)
            print(f"  Test Accuracy: {acc:.4f} | Test F1: {f1:.4f} | Test EER: {eer:.4f}")


        # final summary
        print(f"\n{'=' * 60}")
        print(f"SUMMARY over {len(runs)} runs")
        print(f"{'=' * 60}")


        metrics = {
            "Threshold": all_thrs,
            "Val Accuracy": val_accs,
            "Val F1-Score": val_f1s,
            "Val EER": val_eers,
            "Test Accuracy": test_accs,
            "Test F1-Score": test_f1s,
            "Test EER": test_eers,
            "Test TAR": test_tars,
            "Test FAR": test_fars,
            "Test AUC": test_aucs,
            "Test TAR@FAR1%": test_tarfar1s,
        }

        model_name = experiment_name[12:]
        summary = {
            name: f"{np.mean(values):.4f} ± {np.std(values):.4f}"
            for name, values in metrics.items()
            if name != "Threshold" and "F1-Score" not in name
        }

        # Add model name as the first column
        summary = {"Model": model_name, **summary}
        all_summaries.append(summary)

    df_markdown = pd.DataFrame(all_summaries)
    print(df_markdown.to_markdown(index=False))