import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, SEED


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_split_indices(outdir: str, train_idx: np.ndarray, test_idx: np.ndarray):
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


def prepare_variant_inputs(
    variant_name: str,
    processed_csv_path: str,
    out_base: str = PROCESSED_DIR,
    test_size: float = 0.2,
    seed: int = SEED
) -> Dict:
    
    outdir = os.path.join(out_base, variant_name)
    ensure_dir(outdir)

    if not os.path.isfile(processed_csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv_path}")

    df = pd.read_csv(processed_csv_path)

    if "target" in df.columns:
        y = df["target"].copy()
        X = df.drop(columns=["target"])
    else:
        raise RuntimeError(f"'target' column not found in processed CSV: {processed_csv_path}")

    stratify_vals = y if y.nunique() > 1 else None

    if stratify_vals is not None:
        X_train, X_test, y_train, y_test, train_idx, test_idx = _stratified_indices_split(X, y, test_size=test_size, seed=seed)
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = _random_indices_split(X, y, test_size=test_size, seed=seed)

    X_train_path = os.path.join(outdir, f"{variant_name}_X_train.csv")
    X_test_path = os.path.join(outdir, f"{variant_name}_X_test.csv")
    y_train_path = os.path.join(outdir, f"{variant_name}_y_train.csv")
    y_test_path = os.path.join(outdir, f"{variant_name}_y_test.csv")

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

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


def _stratified_indices_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:

    rng = np.random.RandomState(seed)
    n = len(X)
    indices = np.arange(n)

    classes = {}
    for idx, label in zip(indices, y.values):
        classes.setdefault(label, []).append(idx)

    train_idx_list = []
    test_idx_list = []

    for label, idxs in classes.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_test = int(np.round(len(idxs) * test_size))
        if n_test >= len(idxs):
            n_test = max(1, len(idxs) - 1)
        test_idxs = idxs[:n_test]
        train_idxs = idxs[n_test:]
        train_idx_list.append(train_idxs)
        test_idx_list.append(test_idxs)

    if train_idx_list:
        train_idx = np.concatenate(train_idx_list)
    else:
        train_idx = np.array([], dtype=int)
    if test_idx_list:
        test_idx = np.concatenate(test_idx_list)
    else:
        test_idx = np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def _random_indices_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Random split (no stratification) using numpy RNG.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(np.round(n * test_size))
    if n_test >= n:
        n_test = max(1, n - 1)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def prepare_all_variants(
    out_base: str = PROCESSED_DIR,
    variants: List[Dict] = PREPROCESSING_VARIANTS,
    test_size: float = 0.2,
    seed: int = SEED
) -> List[Dict]:
    summaries = []
    for var in variants:
        variant_name = var["name"]
        processed_csv = os.path.join(out_base, variant_name, f"{variant_name}_processed.csv")
        summary = prepare_variant_inputs(variant_name, processed_csv, out_base=out_base, test_size=test_size, seed=seed)
        summaries.append(summary)
    return summaries