"""
src/evaluate.py
Train final KNN models per preprocessing variant using a chosen K (automatically selected
from aggregated CV results if available) and evaluate on reserved test sets.

Outputs per variant:
- confusion matrix image: reports/figs/<variant>_confusion_matrix.png
- metrics CSV appended to reports/results/final_metrics.csv with columns:
  preprocessing_variant, K, accuracy, precision, recall, f1, n_test
- optional per-variant pickle of the fitted model in RESULTS_DIR/<variant>_model.pkl
"""

import os
import pickle
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, RESULTS_DIR, FIGS_DIR, SEED, PRIMARY_METRIC

sns.set(style="whitegrid", context="talk")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_split_csvs(variant_name: str, out_base: str = PROCESSED_DIR) -> Dict[str, str]:
    """
    Return paths for X_train, X_test, y_train, y_test CSVs for the variant.
    Raises FileNotFoundError if expected files are missing.
    """
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
    """
    Choose candidate best K for a variant. Strategy:
    1) If agg_csv_path (reports/results/knn_cv_agg.csv) exists, pick the K with highest mean_metric;
       tie-breaker: smaller std_metric.
    2) Otherwise, pick median of provided k_grid or raise.
    """
    if agg_csv_path and os.path.isfile(agg_csv_path):
        agg = pd.read_csv(agg_csv_path)
        group = agg[agg["preprocessing_variant"] == variant_name]
        if group.empty:
            raise RuntimeError(f"No aggregated CV results found for variant: {variant_name}")
        best = group.sort_values(["mean_metric", "std_metric"], ascending=[False, True]).iloc[0]
        return int(best["K"])
    if k_grid:
        # default fallback: choose middle K
        return int(k_grid[len(k_grid) // 2])
    raise RuntimeError("Cannot choose K: no agg_csv and no k_grid provided.")


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, outpath: str, variant_name: str, k: int):
    ensure_dir(os.path.dirname(outpath) or ".")
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{variant_name}  K={k}  Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def evaluate_variants(agg_csv_path: str = os.path.join(RESULTS_DIR, "knn_cv_agg.csv"), save_models: bool = True) -> List[Dict[str, Any]]:
    """
    For each variant in PREPROCESSING_VARIANTS:
    - Load train/test splits
    - Choose K (from agg_csv_path or fallback)
    - Train KNN on full training set
    - Evaluate on test set: compute accuracy, precision, recall, f1
    - Save confusion matrix and append metrics to results CSV
    Returns list of per-variant result dicts.
    """
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGS_DIR)

    final_metrics_path = os.path.join(RESULTS_DIR, "final_metrics.csv")
    results_rows = []

    for var in PREPROCESSING_VARIANTS:
        variant_name = var["name"]
        print(f"Evaluating variant: {variant_name}")

        # Load splits
        paths = _load_split_csvs(variant_name)
        X_train = pd.read_csv(paths["X_train"]).values
        X_test = pd.read_csv(paths["X_test"]).values
        y_train = pd.read_csv(paths["y_train"]).squeeze().values
        y_test = pd.read_csv(paths["y_test"]).squeeze().values

        # Choose K
        try:
            chosen_k = _choose_k_for_variant(variant_name, agg_csv_path=agg_csv_path, k_grid=None)
        except Exception:
            # fallback: choose 5
            chosen_k = 5

        # Train KNN on full training set with chosen_k
        model = KNeighborsClassifier(n_neighbors=chosen_k, n_jobs=-1)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        # Save confusion matrix figure
        cm_path = os.path.join(FIGS_DIR, f"{variant_name}_confusion_matrix.png")
        _plot_confusion(y_test, y_pred, cm_path, variant_name, chosen_k)

        # Optionally save model
        if save_models:
            model_path = os.path.join(RESULTS_DIR, f"{variant_name}_knn_k{chosen_k}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            model_path = None

        row = {
            "preprocessing_variant": variant_name,
            "K": int(chosen_k),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_test": int(len(y_test)),
            "confusion_matrix_png": cm_path,
            "model_pickle": model_path
        }
        results_rows.append(row)

    # Append or create final_metrics.csv
    df_final = pd.DataFrame(results_rows)
    if os.path.isfile(final_metrics_path):
        existing = pd.read_csv(final_metrics_path)
        df_final = pd.concat([existing, df_final], ignore_index=True)
    df_final.to_csv(final_metrics_path, index=False)
    print("Saved final metrics to:", final_metrics_path)
    return results_rows
