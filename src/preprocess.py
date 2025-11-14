"""
src/preprocess.py
Apply preprocessing variants defined in src.config to the raw DataFrame and
save processed CSVs, transformer artifacts (pickles), and a short preprocessing log.

This version includes sklearn compatibility for OneHotEncoder (sparse vs sparse_output)
and robust handling of numeric/binary/categorical column detection.
"""

import os
import pickle
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, RAW_CSV, SEED


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _save_log(lines: List[str], path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def _detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Heuristic detection of numeric, binary, and categorical columns.
    Returns (numeric_cols, binary_cols, categorical_cols).
    """
    # Numeric dtype columns (includes binary numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Non-numeric columns (likely categorical)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Identify numeric binary columns: numeric columns with exactly two unique non-null values
    binary_cols = []
    for col in numeric_cols[:]:  # iterate over a copy
        uniques = pd.Series(df[col].dropna().unique())
        if uniques.nunique() == 2:
            binary_cols.append(col)

    # Treat binary numeric columns as binary (preserve numeric 0/1), so remove them from numeric_cols
    numeric_cols = [c for c in numeric_cols if c not in binary_cols]

    return numeric_cols, binary_cols, categorical_cols


def _build_transformers(numeric_cols: List[str], categorical_cols: List[str], variant: Dict):
    """
    Build sklearn transformers (imputers, encoders, scalers) according to variant dict.
    Returns (preprocessor, artifacts) where artifacts is a dict of fitted transformer objects
    (imputer_num, imputer_cat, encoder, scaler) for saving.
    """
    # numeric imputer
    num_strategy = variant.get("numeric_imputation", "mean")
    imputer_num = SimpleImputer(strategy=num_strategy)

    # categorical imputer: use most_frequent (mode) or fallback to constant
    cat_strategy = variant.get("categorical_imputation", "mode")
    if cat_strategy == "mode":
        imputer_cat = SimpleImputer(strategy="most_frequent")
    else:
        imputer_cat = SimpleImputer(strategy="constant", fill_value="MISSING")

    # OneHotEncoder compatibility across sklearn versions
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    # scaler selection
    scaler_name = variant.get("scaler", "standard")
    scaler = MinMaxScaler() if scaler_name == "minmax" else StandardScaler()

    # Pipelines
    numeric_pipeline = Pipeline([("imputer", imputer_num), ("scaler", scaler)])
    categorical_pipeline = Pipeline([("imputer", imputer_cat), ("onehot", encoder)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    artifacts = {
        "imputer_num": imputer_num,
        "imputer_cat": imputer_cat,
        "encoder": encoder,
        "scaler": scaler,
    }

    return preprocessor, artifacts


def apply_variant_and_save(df_raw: pd.DataFrame, variant: Dict, out_base: str = PROCESSED_DIR) -> Dict:
    """
    Apply preprocessing specified in variant to df_raw and save:
    - processed CSV: out_base/<variant_name>/<variant_name>_processed.csv
    - pickled artifacts in same folder
    - small log file listing operations and parameters

    Returns a summary dict with paths and metadata.
    """
    variant_name = variant["name"]
    outdir = os.path.join(out_base, variant_name)
    ensure_dir(outdir)

    # Work on a deep copy of raw DataFrame
    df = df_raw.copy(deep=True)

    # Detect columns
    numeric_cols, binary_cols, categorical_cols = _detect_columns(df)

    # If categorical columns overlap with numeric (rare), ensure categorical_cols only contains non-numeric columns
    categorical_cols = [c for c in categorical_cols if c not in numeric_cols and c not in binary_cols]

    # Build transformers
    preprocessor, artifacts = _build_transformers(numeric_cols, categorical_cols, variant)

    # Columns to feed into transformer
    transformer_cols = numeric_cols + categorical_cols
    if len(transformer_cols) == 0 and len(binary_cols) == 0:
        raise RuntimeError("No columns detected for transformation. Check dataframe dtypes and content.")

    # Prepare X for fitting/transforming (only transformer_cols; binary cols and target are preserved later)
    X = df[transformer_cols].copy() if transformer_cols else pd.DataFrame(index=df.index)

    # Fit transformer (skip fit if transformer has no columns for a given group)
    if transformer_cols:
        preprocessor.fit(X)

        # Transform
        X_trans = preprocessor.transform(X)

        # Build output column names:
        numeric_out = numeric_cols
        cat_feature_names = []
        if categorical_cols:
            # encoder may be nested under the ColumnTransformer; use artifacts["encoder"] to get feature names
            try:
                cat_feature_names = artifacts["encoder"].get_feature_names_out(categorical_cols).tolist()
            except Exception:
                # fallback naming
                cat_feature_names = [f"{c}_{i}" for c in categorical_cols for i in range(1)]
        out_columns = numeric_out + cat_feature_names

        # DataFrame from transformed array
        df_processed = pd.DataFrame(X_trans, columns=out_columns, index=df.index)
    else:
        # No numeric/categorical to transform, create empty processed DF to which we will append preserved cols
        df_processed = pd.DataFrame(index=df.index)

    # Preserve numeric binary columns (keep original 0/1 numeric form)
    for col in binary_cols:
        if col in df.columns:
            df_processed[col] = df[col].values

    # Append target column unchanged if present
    if "target" in df.columns:
        df_processed["target"] = df["target"].values

    # Reorder so that target is last
    cols_order = [c for c in df_processed.columns if c != "target"] + (["target"] if "target" in df_processed.columns else [])
    df_processed = df_processed[cols_order]

    # Save processed CSV
    csv_path = os.path.join(outdir, f"{variant_name}_processed.csv")
    df_processed.to_csv(csv_path, index=False)

    # Save artifacts (pickles)
    _save_pickle(artifacts["imputer_num"], os.path.join(outdir, "imputer_num.pkl"))
    _save_pickle(artifacts["imputer_cat"], os.path.join(outdir, "imputer_cat.pkl"))
    _save_pickle(artifacts["encoder"], os.path.join(outdir, "encoder.pkl"))
    _save_pickle(artifacts["scaler"], os.path.join(outdir, "scaler.pkl"))
    _save_pickle(preprocessor, os.path.join(outdir, "preprocessor.pkl"))

    # Save log
    log_lines = [
        f"variant_name: {variant_name}",
        f"numeric_imputation: {variant.get('numeric_imputation')}",
        f"categorical_imputation: {variant.get('categorical_imputation')}",
        f"scaler: {variant.get('scaler')}",
        f"numeric_columns: {numeric_cols}",
        f"categorical_columns: {categorical_cols}",
        f"binary_columns_preserved: {binary_cols}",
        f"processed_csv: {csv_path}",
        f"artifact_dir: {outdir}",
        f"seed: {SEED}",
    ]
    _save_log(log_lines, os.path.join(outdir, f"{variant_name}_preprocessing_log.txt"))

    summary = {
        "variant": variant_name,
        "csv": csv_path,
        "artifact_dir": outdir,
        "log": os.path.join(outdir, f"{variant_name}_preprocessing_log.txt"),
        "n_rows": int(df_processed.shape[0]),
        "n_columns": int(df_processed.shape[1]),
    }
    return summary


def run_all_variants(csv_path: str = RAW_CSV, variants: List[Dict] = PREPROCESSING_VARIANTS, out_base: str = PROCESSED_DIR) -> List[Dict]:
    """
    Load raw CSV and run apply_variant_and_save for each variant.
    Returns list of summary dicts for each variant.
    """
    df_raw = pd.read_csv(csv_path)
    summaries = []
    for var in variants:
        summary = apply_variant_and_save(df_raw, var, out_base=out_base)
        summaries.append(summary)
    return summaries
