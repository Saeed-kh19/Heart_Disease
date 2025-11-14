"""
run_pipeline.py
End-to-end pipeline runner. Run from project root where src/ is importable.

Sequence:
1. Preprocess: create processed variants
2. Prepare inputs: create stratified train/test splits
3. Tune: cross-validation grid search and save CV results
4. Evaluate: train final models on train splits and evaluate on reserved test sets
5. Sensitivity: run selected sensitivity checks
6. Produce final report placeholder check (ensures expected files exist)
"""

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
    # final metrics and CV agg
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
        # do not fail hard here; allow user to inspect logs
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
