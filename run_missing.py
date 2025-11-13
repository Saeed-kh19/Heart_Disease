"""
run_missing.py
Run only the missing-value report and heuristic assessment.
"""

from src.load_data import load_data
from src.missing import save_missing_report, heuristic_missingness_assessment

CSV_PATH = "data/raw/heart_dataset.csv"
OUTDIR = "reports/"

def main():
    df = load_data(CSV_PATH)
    csv_path = save_missing_report(df, OUTDIR)
    note = heuristic_missingness_assessment(df, target_col="target", threshold_pct=5.0)
    with open(OUTDIR + "missing_assessment.txt", "w", encoding="utf-8") as f:
        f.write(note + "\n")
    print("Missing report saved to:", csv_path)
    print("Missingness assessment:", note)

if __name__ == "__main__":
    main()
