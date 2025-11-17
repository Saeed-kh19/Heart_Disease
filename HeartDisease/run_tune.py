import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.tune_cv import run_grid
from src.config import RESULTS_DIR, FIGS_DIR, K_GRID

sns.set(style="whitegrid", context="talk")

def plot_metric_vs_k(agg_df: pd.DataFrame, outpath: str):

    plt.figure(figsize=(8, 5))
    for variant, group in agg_df.groupby("preprocessing_variant"):
        group_sorted = group.sort_values("K")
        plt.errorbar(group_sorted["K"], group_sorted["mean_metric"], yerr=group_sorted["std_metric"], label=variant, marker="o", capsize=4)
    plt.xlabel("K (neighbors)")
    plt.ylabel("Mean " + "F1")
    plt.title("KNN CV mean F1 vs K per preprocessing variant")
    plt.xticks(K_GRID)
    plt.legend()
    plt.tight_layout()
    ensure_dir = lambda p: os.makedirs(os.path.dirname(p), exist_ok=True)
    ensure_dir(outpath)
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    res = run_grid()
    agg_df = res["agg"]
    outplot = os.path.join(FIGS_DIR, "knn_cv_metric_vs_k.png")
    plot_metric_vs_k(agg_df, outplot)
    print("Saved CV metric vs K plot to:", outplot)
    candidates = {}
    for variant, group in agg_df.groupby("preprocessing_variant"):
        best_idx = group.sort_values(["mean_metric", "std_metric"], ascending=[False, True]).iloc[0]
        candidates[variant] = {"K": int(best_idx["K"]), "mean": float(best_idx["mean_metric"]), "std": float(best_idx["std_metric"])}
    print("Candidate best K per variant (by mean metric, tie-broken by std):")
    for v, info in candidates.items():
        print(f"- {v}: K={info['K']}, mean={info['mean']:.4f}, std={info['std']:.4f}")

if __name__ == "__main__":
    main()
