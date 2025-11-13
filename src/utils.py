import os
import io
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_initial_audit(df, outdir, target_col="target"):
    ensure_dir(outdir)

    head_text = df.head(10).to_csv(index=False)
    save_text(os.path.join(outdir, "head.txt"), head_text)

    buf = io.StringIO()
    df.info(buf=buf)
    info_text = buf.getvalue()
    save_text(os.path.join(outdir, "info.txt"), info_text)

    describe_text = df.describe(include="all").to_string()
    save_text(os.path.join(outdir, "describe.txt"), describe_text)

    if target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False).sort_index()
        percents = df[target_col].value_counts(normalize=True, dropna=False).sort_index() * 100
        lines = ["Value\tCount\tPercent"]
        for val in counts.index:
            lines.append(f"{val}\t{counts.loc[val]}\t{percents.loc[val]:.2f}%")
        save_text(os.path.join(outdir, "class_distribution.txt"), "\n".join(lines))
    else:
        save_text(
            os.path.join(outdir, "class_distribution.txt"),
            f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}",
        )
