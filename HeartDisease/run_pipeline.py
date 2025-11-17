import sys
import os
from pathlib import Path
import subprocess

STEPS = [
    ("Preprocess", ["python", "run_preprocess.py"]),
    ("Prepare inputs", ["python", "run_prepare_inputs.py"]),
    ("Tune CV", ["python", "run_tune.py"]),
    ("Evaluate", ["python", "run_evaluate.py"]),
    ("Sensitivity", ["python", "run_sensitivity.py"]),
]

def run_step(name, cmd):
    print(f"=== Step: {name} ===")
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"Step '{name}' failed with exit code {rc}")

def check_expected_outputs():
    errs = []
    expected = [
        "reports/results/final_metrics.csv",
        "reports/results/knn_cv_agg.csv",
        "reports/figs/knn_cv_metric_vs_k.png",
    ]
    for p in expected:
        if not Path(p).exists():
            errs.append(p)
    if errs:
        print("Warning: expected outputs missing:")
        for e in errs:
            print(" -", e)
    else:
        print("All key outputs present.")

def main():
    
    try:
        for name, cmd in STEPS:
            run_step(name, cmd)
        check_expected_outputs()
        print("Pipeline finished successfully.")
    except Exception as e:
        print("Pipeline failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
