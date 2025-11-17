import os
import pickle
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, RESULTS_DIR, FIGS_DIR, SEED, PRIMARY_METRIC

sns.set(style="whitegrid", context="talk")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_split_csvs(variant_name: str, out_base: str = PROCESSED_DIR) -> Dict[str, str]:
    base = os.path.join(out_base, variant_name)
    X_train = os.path.join(base, f"{variant_name}_X_train.csv")
    X_test = os.path.join(base, f"{variant_name}_X_test.csv")
    y_train = os.path.join(base, f"{variant_name}_y_train.csv")
    y_test = os.path.join(base, f"{variant_name}_y_test.csv")
    for p in (X_train, X_test, y_train, y_test):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required split file missing: {p}")
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def _choose_k_for_variant(variant_name: str, agg_csv_path: str = None, k_grid: List[int] = None) -> int:
    if agg_csv_path and os.path.isfile(agg_csv_path):
        agg = pd.read_csv(agg_csv_path)
        group = agg[agg["preprocessing_variant"] == variant_name]
        if group.empty:
            raise RuntimeError(f"No aggregated CV results found for variant: {variant_name}")
        best = group.sort_values(["mean_metric", "std_metric"], ascending=[False, True]).iloc[0]
        return int(best["K"])
    if k_grid:
        return int(k_grid[len(k_grid) // 2])
    raise RuntimeError("Cannot choose K: no agg_csv and no k_grid provided.")


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm, labels = _confusion_matrix(y_true, y_pred)
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "precision_weighted": float(np.average(precisions, weights=cm.sum(axis=1))),
        "recall_weighted": float(np.average(recalls, weights=cm.sum(axis=1))),
        "f1_weighted": float(np.average(f1s, weights=cm.sum(axis=1))),
    }


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, outpath: str, variant_name: str, k: int):
    ensure_dir(os.path.dirname(outpath) or ".")
    cm, labels = _confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{variant_name}  K={k}  Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:

    n_test = X_test.shape[0]
    preds = np.empty(n_test, dtype=y_train.dtype)
    for i in range(n_test):
        q = X_test[i]
        dists = np.linalg.norm(X_train - q, axis=1)
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]
        unique, counts = np.unique(nn_labels, return_counts=True)
        max_count = counts.max()
        candidates = unique[counts == max_count]
        if len(candidates) == 1:
            preds[i] = candidates[0]
        else:
            best_candidate = None
            best_avg_dist = None
            for cand in candidates:
                cand_mask = nn_labels == cand
                avgd = dists[nn_idx][cand_mask].mean()
                if (best_avg_dist is None) or (avgd < best_avg_dist):
                    best_avg_dist = avgd
                    best_candidate = cand
            preds[i] = best_candidate
    return preds


def evaluate_variants(agg_csv_path: str = os.path.join(RESULTS_DIR, "knn_cv_agg.csv"), save_models: bool = True) -> List[Dict[str, Any]]:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGS_DIR)

    final_metrics_path = os.path.join(RESULTS_DIR, "final_metrics.csv")
    results_rows = []

    for var in PREPROCESSING_VARIANTS:
        variant_name = var["name"]
        print(f"Evaluating variant: {variant_name}")

        paths = _load_split_csvs(variant_name)
        X_train = pd.read_csv(paths["X_train"]).values
        X_test = pd.read_csv(paths["X_test"]).values
        y_train = pd.read_csv(paths["y_train"]).squeeze().values
        y_test = pd.read_csv(paths["y_test"]).squeeze().values

        try:
            chosen_k = _choose_k_for_variant(variant_name, agg_csv_path=agg_csv_path, k_grid=None)
        except Exception:
            chosen_k = 5

        y_pred = _knn_predict(X_train, y_train, X_test, chosen_k)

        metrics = _precision_recall_f1(y_test, y_pred)
        if PRIMARY_METRIC == "f1":
            report_f1 = metrics["f1_macro"]
        elif PRIMARY_METRIC == "f1_weighted":
            report_f1 = metrics["f1_weighted"]
        elif PRIMARY_METRIC == "f1_macro":
            report_f1 = metrics["f1_macro"]
        else:
            report_f1 = metrics["f1_macro"]

        acc = metrics["accuracy"]
        prec = metrics["precision_macro"]
        rec = metrics["recall_macro"]
        f1 = report_f1

        cm_path = os.path.join(FIGS_DIR, f"{variant_name}_confusion_matrix.png")
        _plot_confusion(y_test, y_pred, cm_path, variant_name, chosen_k)

        if save_models:
            model_path = os.path.join(RESULTS_DIR, f"{variant_name}_knn_k{chosen_k}_model.pkl")
            model_obj = {"X_train": X_train, "y_train": y_train, "k": int(chosen_k)}
            with open(model_path, "wb") as f:
                pickle.dump(model_obj, f)
        else:
            model_path = None

        row = {
            "preprocessing_variant": variant_name,
            "K": int(chosen_k),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "n_test": int(len(y_test)),
            "confusion_matrix_png": cm_path,
            "model_pickle": model_path
        }
        results_rows.append(row)

    df_final = pd.DataFrame(results_rows)
    if os.path.isfile(final_metrics_path):
        existing = pd.read_csv(final_metrics_path)
        df_final = pd.concat([existing, df_final], ignore_index=True)
    df_final.to_csv(final_metrics_path, index=False)
    print("Saved final metrics to:", final_metrics_path)
    return results_rows
