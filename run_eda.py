"""
run_eda.py
Example script that runs EDA functions on the raw dataset and writes short observations.
Run from project root (where src/ is importable).
"""
import os
from src.load_data import load_data
import src.eda as eda

# Paths (adjust if your project root differs)
CSV_PATH = "data/raw/heart_dataset.csv"
FIG_DIR = "reports/figs"
REPORT_SUMMARY = "reports/eda_summary.txt"

# Select features for EDA
NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "target"]

def write_observations(obs_lines, outpath):
    """Write observation lines to a summary text file overwriting any existing file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for line in obs_lines:
            f.write(line + "\n")

def main():
    df = load_data(CSV_PATH)

    # Histograms
    hist_paths = eda.plot_histograms(df, NUMERICAL, FIG_DIR)
    # Boxplots
    box_paths = eda.plot_boxplots(df, NUMERICAL, FIG_DIR)
    # Q-Q plots for two skewed features (choose chol and oldpeak as likely skewed)
    qq1 = eda.plot_qq(df, "chol", FIG_DIR)
    qq2 = eda.plot_qq(df, "oldpeak", FIG_DIR)
    # Correlation heatmap
    corr = eda.plot_corr_heatmap(df, NUMERICAL + ["target"], FIG_DIR, annot=True)
    # Countplots
    counts = eda.plot_countplots(df, CATEGORICAL, FIG_DIR)
    # Optional pairplot (small subset)
    pair = None
    try:
        pair = eda.plot_pairplot(df, ["age", "thalach", "oldpeak"], FIG_DIR, hue="target")
    except Exception:
        # Pairplot may fail on small machines or with many NA rows; ignore and continue
        pair = None

    # Simple observations (examples). You should refine these after inspecting the figures.
    observations = [
        "Histograms: age shows a roughly normal distribution skewed slightly older; chol and oldpeak show right skew.",
        "Boxplots: chol and oldpeak contain clear high-value outliers worth checking before imputation.",
        "Q-Q plots: chol deviates from normal in the right tail; oldpeak shows substantial non-normality.",
        "Correlation heatmap: thalach and age show moderate correlation patterns with target; check exact values in heatmap.",
        "Countplots: class imbalance visible if one target class dominates; also validate sex and cp distributions.",
        "Pairplot: relationships between age, thalach, and oldpeak differ between target classes (inspect saved figure)."
    ]

    write_observations(observations, REPORT_SUMMARY)
    print("EDA finished. Figures saved to:", FIG_DIR)
    print("EDA summary saved to:", REPORT_SUMMARY)

if __name__ == "__main__":
    main()
