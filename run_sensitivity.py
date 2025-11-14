"""
run_sensitivity.py
Runner to execute the sensitivity checks implemented in src/sensitivity.py.

Produces:
- reports/results/sensitivity_comparison.txt
- reports/results/scaler_impact.txt
- reports/figs/scaler_distance_change.png
"""

from src.sensitivity import compare_variants, analyze_scaler_impact

def main():
    para = compare_variants()
    print("Variant comparison paragraph saved.")
    print(para)
    res = analyze_scaler_impact()
    print("Scaler impact analysis saved to:", res["txt"])
    print("Scaler distance plot saved to:", res["plot"])

if __name__ == "__main__":
    main()
