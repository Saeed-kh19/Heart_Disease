"""
src/tune_cv.py
Run K-NN cross-validation grid search per preprocessing variant and record results.

Outputs:
- results CSV per run: reports/results/knn_cv_results.csv
  columns: preprocessing_variant, K, fold, metric_value
- aggregated CSV: reports/results/knn_cv_agg.csv
  columns: preprocessing_variant, K, mean_metric, std_metric
- Pickle of full results dict: reports/results/knn_cv_results.pkl
- The module exposes run_grid(...) to be called from a runner.
"""

import os
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, RESULTS_DIR, K_GRID, CV_FOLDS, PRIMARY_METRIC, SEED

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _load_processed_csv(variant_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, variant_name, f"{variant_name}_processed.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Processed CSV not found for variant {variant_name}: {path}")
    return pd.read_csv(path)

def _metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Currently supports PRIMARY_METRIC == "f1", "f1_macro", "f1_weighted"
    if PRIMARY_METRIC == "f1":
        return f1_score(y_true, y_pred)
    elif PRIMARY_METRIC == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")
    elif PRIMARY_METRIC == "f1_weighted":
        return f1_score(y_true, y_pred, average="weighted")
    else:
        # fallback to binary F1
        return f1_score(y_true, y_pred)

def run_grid(k_grid: List[int] = K_GRID, cv_folds: int = CV_FOLDS, variants: List[Dict] = PREPROCESSING_VARIANTS, seed: int = SEED) -> Dict[str, Any]:
    """
    Run stratified K-fold CV for each variant and each K in k_grid.
    Returns a dictionary with detailed per-fold results and saves CSVs/plots externally.
    """
    results = []  # rows: dicts with variant, K, fold, metric_value
    summaries = []  # aggregated per variant/K

    rng = np.random.RandomState(seed)

    for var in variants:
        variant_name = var["name"]
        print(f"Processing variant: {variant_name}")
        df = _load_processed_csv(variant_name)

        if "target" not in df.columns:
            raise RuntimeError(f"Processed CSV for {variant_name} missing 'target' column")

        y = df["target"].values
        X = df.drop(columns=["target"]).values

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        for K in k_grid:
            fold_idx = 0
            fold_scores = []
            for train_idx, test_idx in skf.split(X, y):
                fold_idx += 1
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = _metric_fn(y_test, y_pred)

                results.append({
                    "preprocessing_variant": variant_name,
                    "K": int(K),
                    "fold": int(fold_idx),
                    "metric_value": float(score)
                })
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            summaries.append({
                "preprocessing_variant": variant_name,
                "K": int(K),
                "mean_metric": mean_score,
                "std_metric": std_score
            })

    # Save detailed and aggregated CSVs + pickle
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
