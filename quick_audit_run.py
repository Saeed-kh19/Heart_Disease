# quick_audit_run.py (example script you can run once the repo has code)
from src.load_data import load_data
from src.utils import save_initial_audit

# Adjust these paths for your environment
csv_path = "data/raw/heart_dataset.csv"
reports_dir = "reports/"

# Load raw dataset into df_raw
df_raw = load_data(csv_path)

# Save initial audit outputs into reports/
save_initial_audit(df_raw, reports_dir, target_col="target")
print("Initial audit saved to reports/ (head.txt, info.txt, describe.txt, class_distribution.txt)")
