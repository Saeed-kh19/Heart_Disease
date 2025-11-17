import os
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def missing_table(df: pd.DataFrame) -> pd.DataFrame:

    total = len(df)
    cols = []
    for col in df.columns:
        miss = df[col].isna().sum()
        pct = (miss / total) * 100
        cols.append({"column": col, "missing_count": int(miss), "missing_percent": round(pct, 3)})
    table = pd.DataFrame(cols).sort_values("missing_percent", ascending=False).reset_index(drop=True)
    return table

def save_missing_report(df: pd.DataFrame, outdir: str, filename: str = "missing_report.csv"):

    ensure_dir(outdir)
    table = missing_table(df)
    csv_path = os.path.join(outdir, filename)
    txt_path = os.path.join(outdir, filename.replace(".csv", ".txt"))

    # Save CSV
    table.to_csv(csv_path, index=False)

    # Save readable text summary (one line per column)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("column,missing_count,missing_percent\n")
        for _, row in table.iterrows():
            f.write(f"{row['column']},{row['missing_count']},{row['missing_percent']}%\n")

    return csv_path

def heuristic_missingness_assessment(df: pd.DataFrame, target_col: str = "target", threshold_pct: float = 5.0) -> str:

    table = missing_table(df)
    if table["missing_count"].sum() == 0:
        return "No missing values detected."

    high = table[table["missing_percent"] > threshold_pct]
    total_missing = table["missing_count"].sum()

    # Check if missingness correlates with target (simple check: compare mean % missing by target class)
    if target_col in df.columns:
        by_target = []
        for val in sorted(df[target_col].dropna().unique()):
            subset = df[df[target_col] == val]
            pct_missing = subset.isna().sum().sum() / (len(subset) * len(df.columns)) * 100 if len(subset) > 0 else 0
            by_target.append((val, round(pct_missing, 3)))
        target_note = "Missingness by target sample-wide percent: " + "; ".join(f"{v}:{p}%" for v, p in by_target)
    else:
        target_note = "Target column not found; cannot compare missingness by class."

    if len(high) == 0:
        return f"Missing values present but all columns <= {threshold_pct}% missing (likely low or random). {target_note}"
    elif len(high) <= 3:
        cols = ", ".join(high["column"].tolist())
        return f"Missingness concentrated in columns: {cols} (each > {threshold_pct}%). {target_note}"
    else:
        cols = ", ".join(high["column"].tolist())
        return f"Multiple columns ({len(high)}) have > {threshold_pct}% missing: {cols}. Investigate systematic causes. {target_note}"
