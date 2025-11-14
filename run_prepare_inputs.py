"""
run_prepare_inputs.py
Runner to create stratified train/test splits for all processed variants.
Run this from the project root (where src/ is importable).
"""

from src.prepare_inputs import prepare_all_variants
from src.config import PREPROCESSING_VARIANTS, PROCESSED_DIR

def main():
    summaries = prepare_all_variants(out_base=PROCESSED_DIR, variants=PREPROCESSING_VARIANTS, test_size=0.2)
    print("Prepared modelling inputs for variants:")
    for s in summaries:
        print(f"- {s['variant']}: n_train={s['n_train']}, n_test={s['n_test']}, X_train={s['X_train']}, y_train={s['y_train']}")

if __name__ == "__main__":
    main()
