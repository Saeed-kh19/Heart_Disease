import os
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, RESULTS_DIR, K_GRID, CV_FOLDS, PRIMARY_METRIC, SEED


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_processed_csv(variant_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, variant_name, f"{variant_name}_processed.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Processed CSV not found for variant {variant_name}: {path}")
    return pd.read_csv(path)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def _precision_recall_f1_from_cm(cm: np.ndarray) -> Dict[str, float]:
    supports = cm.sum(axis=1)
    precisions = []
    recalls = []
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    total = supports.sum()
    weighted_f1 = float(np.average(f1s, weights=supports)) if total > 0 else 0.0
    weighted_prec = float(np.average(precisions, weights=supports)) if total > 0 else 0.0
    weighted_rec = float(np.average(recalls, weights=supports)) if total > 0 else 0.0
    return {
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "precision_weighted": weighted_prec,
        "recall_weighted": weighted_rec,
        "f1_weighted": weighted_f1,
    }


def _metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm, _ = _confusion_matrix(y_true, y_pred)
    stats = _precision_recall_f1_from_cm(cm)
    if PRIMARY_METRIC == "f1":
        if cm.shape[0] == 2:
            return stats["f1_weighted"]  # for binary weighted == binary f1
        return stats["f1_macro"]
    elif PRIMARY_METRIC == "f1_macro":
        return stats["f1_macro"]
    elif PRIMARY_METRIC == "f1_weighted":
        return stats["f1_weighted"]
    else:
        return stats["f1_macro"]



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
            best_avg = None
            for cand in candidates:
                mask = nn_labels == cand
                avgd = dists[nn_idx][mask].mean()
                if (best_avg is None) or (avgd < best_avg):
                    best_avg = avgd
                    best_candidate = cand
            preds[i] = best_candidate
    return preds



def _stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:

    rng = np.random.RandomState(seed)
    n = len(y)
    labels = np.array(y)
    unique_labels = np.unique(labels)
    folds = [[] for _ in range(n_splits)]
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        rng.shuffle(idxs)
        sizes = [len(idxs) // n_splits + (1 if i < (len(idxs) % n_splits) else 0) for i in range(n_splits)]
        start = 0
        for fold_i, sz in enumerate(sizes):
            if sz > 0:
                folds[fold_i].extend(idxs[start:start+sz].tolist())
            start += sz
    pairs = []
    for i in range(n_splits):
        test_idx = np.array(folds[i], dtype=int)
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=True)
        pairs.append((train_idx, test_idx))
    return pairs



def run_grid(k_grid: List[int] = K_GRID, cv_folds: int = CV_FOLDS,
             variants: List[Dict] = PREPROCESSING_VARIANTS, seed: int = SEED) -> Dict[str, Any]:

    results = []
    summaries = []
    rng = np.random.RandomState(seed)

    for var in variants:
        variant_name = var["name"]
        print(f"Processing variant: {variant_name}")
        df = _load_processed_csv(variant_name)

        if "target" not in df.columns:
            raise RuntimeError(f"Processed CSV for {variant_name} missing 'target' column")

        y = df["target"].values
        X = df.drop(columns=["target"]).values

        fold_pairs = _stratified_kfold_indices(y, n_splits=cv_folds, seed=seed)

        for K in k_grid:
            fold_idx = 0
            fold_scores = []
            for train_idx, test_idx in fold_pairs:
                fold_idx += 1
                if len(train_idx) == 0 or len(test_idx) == 0:
                    # skip degenerate fold
                    continue
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                y_pred = _knn_predict(X_train, y_train, X_test, K)
                score = _metric_fn(y_test, y_pred)

                results.append({
                    "preprocessing_variant": variant_name,
                    "K": int(K),
                    "fold": int(fold_idx),
                    "metric_value": float(score)
                })
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            summaries.append({
                "preprocessing_variant": variant_name,
                "K": int(K),
                "mean_metric": mean_score,
                "std_metric": std_score
            })

    outdir = os.path.join(RESULTS_DIR)
    ensure_dir(outdir)
    results_df = pd.DataFrame(results)
    agg_df = pd.DataFrame(summaries).sort_values(["preprocessing_variant", "K"])

    results_csv = os.path.join(outdir, "knn_cv_results.csv")
    agg_csv = os.path.join(outdir, "knn_cv_agg.csv")
    pkl_path = os.path.join(outdir, "knn_cv_results.pkl")

    results_df.to_csv(results_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    with open(pkl_path, "wb") as f:
        pickle.dump({"detailed": results_df, "agg": agg_df}, f)

    print(f"Saved detailed CV results to: {results_csv}")
    print(f"Saved aggregated CV results to: {agg_csv}")
    print(f"Saved full results pickle to: {pkl_path}")

    return {"detailed": results_df, "agg": agg_df, "paths": {"results_csv": results_csv, "agg_csv": agg_csv, "pkl": pkl_path}}
