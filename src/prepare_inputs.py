"""
src/prepare_inputs.py
Prepare modelling inputs from processed datasets:
- Load each processed CSV under data/processed/<variant_name>/
- Separate X (features) and y (target)
- Perform stratified train/test split (80/20) using SEED from config
- Save train/test CSVs and split index files for reproducibility
"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, SEED

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_split_indices(outdir: str, train_idx: np.ndarray, test_idx: np.ndarray):
    """Save train/test indices as numpy .npy and a small JSON summary for easy inspection."""
    ensure_dir(outdir)
    np.save(os.path.join(outdir, "train_idx.npy"), train_idx)
    np.save(os.path.join(outdir, "test_idx.npy"), test_idx)
    with open(os.path.join(outdir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_idx_path": "train_idx.npy",
            "test_idx_path": "test_idx.npy"
        }, f, indent=2)

def prepare_variant_inputs(variant_name: str, processed_csv_path: str, out_base: str = PROCESSED_DIR, test_size: float = 0.2, seed: int = SEED) -> Dict:
    """
    For a single processed CSV:
    - Load CSV into DataFrame
    - Split into X (all columns except 'target') and y (target)
    - Stratified train/test split (if 'target' present) or random split otherwise
    - Save X_train, X_test, y_train, y_test as CSVs into out_base/<variant_name>/
    - Save train/test indices (npy) and a small JSON summary for reproducibility
    Returns a summary dict with saved paths and sizes.
    """
    outdir = os.path.join(out_base, variant_name)
    ensure_dir(outdir)

    if not os.path.isfile(processed_csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv_path}")

    df = pd.read_csv(processed_csv_path)

    if "target" in df.columns:
        y = df["target"].copy()
        X = df.drop(columns=["target"])
    else:
        # If no target column, raise error because modelling requires target
        raise RuntimeError(f"'target' column not found in processed CSV: {processed_csv_path}")

    # Create stratified split when possible
    stratify_vals = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test, train_idx, test_idx = None, None, None, None, None, None
    if stratify_vals is not None:
        X_train, X_test, y_train, y_test, train_idx, test_idx = _stratified_indices_split(X, y, test_size=test_size, seed=seed)
    else:
        # fallback to random split without stratification
        X_train, X_test, y_train, y_test, train_idx, test_idx = _random_indices_split(X, y, test_size=test_size, seed=seed)

    # Save CSVs
    X_train_path = os.path.join(outdir, f"{variant_name}_X_train.csv")
    X_test_path = os.path.join(outdir, f"{variant_name}_X_test.csv")
    y_train_path = os.path.join(outdir, f"{variant_name}_y_train.csv")
    y_test_path = os.path.join(outdir, f"{variant_name}_y_test.csv")

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    # Save indices
    save_split_indices(outdir, train_idx, test_idx)

    summary = {
        "variant": variant_name,
        "X_train": X_train_path,
        "X_test": X_test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
        "train_idx": os.path.join(outdir, "train_idx.npy"),
        "test_idx": os.path.join(outdir, "test_idx.npy"),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx))
    }
    return summary

def _stratified_indices_split(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Perform a stratified split while returning the train/test row indices relative to X (0-based).
    """
    # Use sklearn train_test_split with return of indices via np.arange
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, idx_train, idx_test

def _random_indices_split(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Random split without stratification; returns indices.
    """
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=None)
    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, idx_train, idx_test

def prepare_all_variants(out_base: str = PROCESSED_DIR, variants: List[Dict] = PREPROCESSING_VARIANTS, test_size: float = 0.2, seed: int = SEED) -> List[Dict]:
    """
    Iterate over variants, find processed CSV under out_base/<variant_name>/<variant_name>_processed.csv,
    and create train/test splits for each. Returns list of summary dicts.
    """
    summaries = []
    for var in variants:
        variant_name = var["name"]
        processed_csv = os.path.join(out_base, variant_name, f"{variant_name}_processed.csv")
        summary = prepare_variant_inputs(variant_name, processed_csv, out_base=out_base, test_size=test_size, seed=seed)
        summaries.append(summary)
    return summaries
