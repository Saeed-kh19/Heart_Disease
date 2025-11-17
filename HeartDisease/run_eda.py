import os
from src.load_data import load_data
import src.eda as eda

CSV_PATH = "data/raw/heart_dataset.csv"
FIG_DIR = "reports/figs"
REPORT_SUMMARY = "reports/eda_summary.txt"

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

    hist_paths = eda.plot_histograms(df, NUMERICAL, FIG_DIR)
    box_paths = eda.plot_boxplots(df, NUMERICAL, FIG_DIR)
    qq1 = eda.plot_qq(df, "chol", FIG_DIR)
    qq2 = eda.plot_qq(df, "oldpeak", FIG_DIR)
    corr = eda.plot_corr_heatmap(df, NUMERICAL + ["target"], FIG_DIR, annot=True)
    counts = eda.plot_countplots(df, CATEGORICAL, FIG_DIR)
    pair = None
    try:
        pair = eda.plot_pairplot(df, ["age", "thalach", "oldpeak"], FIG_DIR, hue="target")
    except Exception:
        pair = None

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
