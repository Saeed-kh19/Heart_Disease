import os
import json
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, RESULTS_DIR, FIGS_DIR

sns.set(style="whitegrid", context="talk")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_final_metrics(path: str = os.path.join(RESULTS_DIR, "final_metrics.csv")):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Final metrics CSV not found: {path}")
    return pd.read_csv(path)


def _load_cv_agg(path: str = os.path.join(RESULTS_DIR, "knn_cv_agg.csv")):
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def compare_variants(out_path: str = os.path.join(RESULTS_DIR, "sensitivity_comparison.txt")) -> str:

    ensure_dir(os.path.dirname(out_path) or ".")
    final = _load_final_metrics()
    agg = _load_cv_agg()

    names = [v["name"] for v in PREPROCESSING_VARIANTS]
    if len(names) < 2:
        raise RuntimeError("Expected at least two preprocessing variants in config")

    vA, vB = names[0], names[1]

    rowA = final[final["preprocessing_variant"] == vA]
    rowB = final[final["preprocessing_variant"] == vB]
    if rowA.empty or rowB.empty:
        raise RuntimeError("Final metrics not present for one or both variants")

    rowA = rowA.sort_values("f1", ascending=False).iloc[0]
    rowB = rowB.sort_values("f1", ascending=False).iloc[0]

    cvA, cvB = None, None
    if agg is not None:
        aggA = agg[agg["preprocessing_variant"] == vA]
        aggB = agg[agg["preprocessing_variant"] == vB]
        try:
            kA = int(rowA["K"])
            kB = int(rowB["K"])
            cvA = aggA[aggA["K"] == kA]
            cvB = aggB[aggB["K"] == kB]
            if not cvA.empty:
                cvA = cvA.iloc[0].to_dict()
            else:
                cvA = None
            if not cvB.empty:
                cvB = cvB.iloc[0].to_dict()
            else:
                cvB = None
        except Exception:
            cvA, cvB = None, None

    para = (
        f"Comparison between {vA} and {vB}: on the reserved test sets {vA} achieved test F1 = "
        f"{rowA['f1']:.4f} (K={int(rowA['K'])}) while {vB} achieved test F1 = {rowB['f1']:.4f} "
        f"(K={int(rowB['K'])})."
    )
    if cvA and cvB:
        para += (
            f" Cross-validation at the chosen K showed mean CV F1 (std) = {cvA['mean_metric']:.4f} "
            f"({cvA['std_metric']:.4f}) for {vA} and {cvB['mean_metric']:.4f} ({cvB['std_metric']:.4f}) for {vB}."
        )
    para += " Based on test F1 and CV stability, prefer the variant with higher test F1 and lower CV std,"
    para += " unless clinical interpretability or simplicity suggests otherwise."

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(para + "\n")

    return para


def _compute_mean_pairwise_distance(X: np.ndarray, sample_frac: float = 1.0, rng_seed: int = 0) -> float:
    """
    Compute mean pairwise Euclidean distance for rows in X. To limit cost, sample rows if sample_frac < 1.
    """
    n = X.shape[0]
    if n == 0:
        return 0.0
    if sample_frac < 1.0 and n > 1000:
        rng = np.random.RandomState(rng_seed)
        idx = rng.choice(n, size=int(n * sample_frac), replace=False)
        Xs = X[idx]
    else:
        Xs = X
    dists = pdist(Xs, metric="euclidean")
    return float(np.mean(dists))


def analyze_scaler_impact(out_txt: str = os.path.join(RESULTS_DIR, "scaler_impact.txt"),
                          out_plot: str = os.path.join(FIGS_DIR, "scaler_distance_change.png")) -> Dict:

    ensure_dir(os.path.dirname(out_txt) or ".")
    ensure_dir(os.path.dirname(out_plot) or ".")

    rows = []
    for var in PREPROCESSING_VARIANTS:
        name = var["name"]
        base = os.path.join(PROCESSED_DIR, name)
        X_train_path = os.path.join(base, f"{name}_X_train.csv")
        preproc_path = os.path.join(base, "preprocessor.pkl")
        scaler_path = os.path.join(base, "scaler.pkl")

        if not os.path.isfile(X_train_path):
            continue

        X_train = pd.read_csv(X_train_path).values

        mean_dist_processed = _compute_mean_pairwise_distance(X_train, sample_frac=0.2, rng_seed=0)

        mean_dist_before = None
        if os.path.isfile(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                try:
                    X_before = scaler.inverse_transform(X_train)
                    mean_dist_before = _compute_mean_pairwise_distance(X_before, sample_frac=0.2, rng_seed=0)
                except Exception:
                    mean_dist_before = None
            except Exception:
                mean_dist_before = None

        rows.append({
            "variant": name,
            "mean_dist_before": mean_dist_before if mean_dist_before is not None else np.nan,
            "mean_dist_after": mean_dist_processed
        })

    df = pd.DataFrame(rows)
    lines = []
    lines.append("Scaler impact summary:")
    for _, r in df.iterrows():
        before = f"{r['mean_dist_before']:.4f}" if not np.isnan(r["mean_dist_before"]) else "N/A"
        after = f"{r['mean_dist_after']:.4f}"
        lines.append(f"- {r['variant']}: mean pairwise distance before scaling = {before}; after scaling = {after}")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["mean_dist_after"], width, label="after scaling")
    if df["mean_dist_before"].notna().any():
        plt.bar(x + width/2, df["mean_dist_before"].fillna(0), width, label="before scaling (approx)")
    plt.xticks(x, df["variant"], rotation=30, ha="right")
    plt.ylabel("Mean pairwise Euclidean distance")
    plt.title("Scaler impact on mean pairwise distances (train set sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    with open(out_txt, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    result = {"table": df, "txt": out_txt, "plot": out_plot}
    return result
