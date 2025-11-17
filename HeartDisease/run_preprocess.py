from src.preprocess import run_all_variants
from src.config import RAW_CSV, PREPROCESSING_VARIANTS, PROCESSED_DIR

def main():
    summaries = run_all_variants(csv_path=RAW_CSV, variants=PREPROCESSING_VARIANTS, out_base=PROCESSED_DIR)
    print("Preprocessing complete. Summaries:")
    for s in summaries:
        print(f"- {s['variant']}: csv={s['csv']}, rows={s['n_rows']}, cols={s['n_columns']}, log={s['log']}")

if __name__ == "__main__":
    main()
