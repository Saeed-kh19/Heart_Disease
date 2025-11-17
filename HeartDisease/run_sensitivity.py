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
