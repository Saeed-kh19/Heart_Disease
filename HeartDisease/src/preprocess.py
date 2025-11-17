import os
import pickle
from typing import List, Dict, Tuple, Any

import pandas as pd
import numpy as np

from .config import PREPROCESSING_VARIANTS, PROCESSED_DIR, RAW_CSV, SEED


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_pickle(obj: Any, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _save_log(lines: List[str], path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def _detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    binary_cols = []
    for col in list(numeric_cols):
        uniques = df[col].dropna().unique()
        if len(uniques) == 2:
            binary_cols.append(col)
    numeric_cols = [c for c in numeric_cols if c not in binary_cols]
    return numeric_cols, binary_cols, categorical_cols



def _fit_numeric_imputer(df: pd.DataFrame, numeric_cols: List[str], strategy: str) -> Dict[str, float]:
    vals = {}
    for c in numeric_cols:
        s = df[c]
        if strategy == "mean":
            vals[c] = float(s.mean(skipna=True)) if s.notna().any() else 0.0
        elif strategy == "median":
            vals[c] = float(s.median(skipna=True)) if s.notna().any() else 0.0
        else:
            # default to mean if unknown
            vals[c] = float(s.mean(skipna=True)) if s.notna().any() else 0.0
    return vals


def _apply_numeric_imputer(arr: pd.DataFrame, imputer_vals: Dict[str, float]) -> pd.DataFrame:
    df = arr.copy()
    for c, v in imputer_vals.items():
        if c in df.columns:
            df[c] = df[c].fillna(v)
    return df


def _fit_categorical_imputer(df: pd.DataFrame, categorical_cols: List[str], strategy: str) -> Dict[str, Any]:
    vals = {}
    for c in categorical_cols:
        s = df[c].dropna()
        if strategy == "mode":
            if not s.empty:
                vals[c] = s.mode().iloc[0]
            else:
                vals[c] = "MISSING"
        else:
            vals[c] = "MISSING"
    return vals


def _apply_categorical_imputer(df: pd.DataFrame, imputer_vals: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for c, v in imputer_vals.items():
        if c in out.columns:
            out[c] = out[c].fillna(v)
    return out


def _fit_onehot_encoder(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, List[Any]]:
    cats = {}
    for c in categorical_cols:
        if c in df.columns:
            uniques = pd.Series(df[c].dropna().unique())
            seen = []
            for val in df[c].astype(object).tolist():
                if pd.isna(val):
                    continue
                if val not in seen:
                    seen.append(val)
            cats[c] = seen
        else:
            cats[c] = []
    return cats


def _transform_onehot(df: pd.DataFrame, categorical_cols: List[str], categories: Dict[str, List[Any]]) -> Tuple[np.ndarray, List[str]]:

    out_cols = []
    arrays = []
    for c in categorical_cols:
        cats = categories.get(c, [])
        colnames = [f"{c}__{str(cat)}" for cat in cats]
        out_cols.extend(colnames)
        if c in df.columns:
            col_vals = df[c].astype(object).values
            mat = np.zeros((len(df), len(cats)), dtype=float)
            for i, val in enumerate(col_vals):
                if pd.isna(val):
                    continue
                try:
                    idx = cats.index(val)
                    mat[i, idx] = 1.0
                except ValueError:
                    pass
        else:
            mat = np.zeros((len(df), len(cats)), dtype=float)
        arrays.append(mat)
    if arrays:
        X_cat = np.hstack(arrays)
    else:
        X_cat = np.zeros((len(df), 0), dtype=float)
    return X_cat, out_cols


def _fit_scaler(df: pd.DataFrame, numeric_cols: List[str], scaler_name: str) -> Dict[str, Any]:
    params = {"type": scaler_name, "params": {}}
    if scaler_name == "minmax":
        mins = {}
        maxs = {}
        for c in numeric_cols:
            s = df[c].astype(float)
            mins[c] = float(s.min(skipna=True)) if s.notna().any() else 0.0
            maxs[c] = float(s.max(skipna=True)) if s.notna().any() else 1.0
            if np.isclose(mins[c], maxs[c]):
                maxs[c] = mins[c] + 1.0
        params["params"] = {"min": mins, "max": maxs}
    else:
        means = {}
        scales = {}
        for c in numeric_cols:
            s = df[c].astype(float)
            means[c] = float(s.mean(skipna=True)) if s.notna().any() else 0.0
            std = float(s.std(skipna=True)) if s.notna().any() else 1.0
            if np.isclose(std, 0.0):
                std = 1.0
            scales[c] = std
        params["params"] = {"mean": means, "scale": scales}
    return params


def _transform_scaler(df: pd.DataFrame, numeric_cols: List[str], scaler_params: Dict[str, Any]) -> np.ndarray:
    if len(numeric_cols) == 0:
        return np.zeros((len(df), 0), dtype=float)
    t = scaler_params.get("type", "standard")
    params = scaler_params.get("params", {})
    X = np.zeros((len(df), len(numeric_cols)), dtype=float)
    for j, c in enumerate(numeric_cols):
        col = df[c].astype(float).values
        if t == "minmax":
            mins = params["min"][c]
            maxs = params["max"][c]
            X[:, j] = (col - mins) / (maxs - mins)
        else:
            mean = params["mean"][c]
            scale = params["scale"][c]
            X[:, j] = (col - mean) / scale
    return X



def apply_variant_and_save(df_raw: pd.DataFrame, variant: Dict, out_base: str = PROCESSED_DIR) -> Dict[str, Any]:

    variant_name = variant["name"]
    outdir = os.path.join(out_base, variant_name)
    ensure_dir(outdir)

    df = df_raw.copy(deep=True)

    numeric_cols, binary_cols, categorical_cols = _detect_columns(df)
    categorical_cols = [c for c in categorical_cols if c not in numeric_cols and c not in binary_cols]

    num_strategy = variant.get("numeric_imputation", "mean")
    cat_strategy = variant.get("categorical_imputation", "mode")
    imputer_num_vals = _fit_numeric_imputer(df, numeric_cols, num_strategy)
    imputer_cat_vals = _fit_categorical_imputer(df, categorical_cols, cat_strategy)

    df_num_imputed = _apply_numeric_imputer(df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index), imputer_num_vals)
    df_cat_imputed = _apply_categorical_imputer(df[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=df.index), imputer_cat_vals)

    encoder_categories = _fit_onehot_encoder(df_cat_imputed, categorical_cols)

    scaler_params = _fit_scaler(df_num_imputed, numeric_cols, variant.get("scaler", "standard"))

    X_num = _transform_scaler(df_num_imputed, numeric_cols, scaler_params)
    X_cat, cat_colnames = _transform_onehot(df_cat_imputed, categorical_cols, encoder_categories)

    out_columns = list(numeric_cols) + cat_colnames
    df_processed = pd.DataFrame(np.hstack([X_num, X_cat]) if (X_num.shape[1] + X_cat.shape[1]) > 0 else np.zeros((len(df), 0)), columns=out_columns, index=df.index)

    for c in binary_cols:
        if c in df.columns:
            df_processed[c] = df[c].values

    if "target" in df.columns:
        df_processed["target"] = df["target"].values

    cols_order = [c for c in df_processed.columns if c != "target"] + (["target"] if "target" in df_processed.columns else [])
    df_processed = df_processed[cols_order]

    csv_path = os.path.join(outdir, f"{variant_name}_processed.csv")
    df_processed.to_csv(csv_path, index=False)

    _save_pickle({"strategy": num_strategy, "values": imputer_num_vals}, os.path.join(outdir, "imputer_num.pkl"))
    _save_pickle({"strategy": cat_strategy, "values": imputer_cat_vals}, os.path.join(outdir, "imputer_cat.pkl"))
    _save_pickle({"categorical_columns": categorical_cols, "categories": encoder_categories}, os.path.join(outdir, "encoder.pkl"))
    _save_pickle({"type": scaler_params["type"], "params": scaler_params["params"]}, os.path.join(outdir, "scaler.pkl"))
    _save_pickle({"numeric_cols": numeric_cols, "binary_cols": binary_cols, "categorical_cols": categorical_cols, "variant": variant}, os.path.join(outdir, "preprocessor.pkl"))

    log_lines = [
        f"variant_name: {variant_name}",
        f"numeric_imputation: {num_strategy}",
        f"categorical_imputation: {cat_strategy}",
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
    df_raw = pd.read_csv(csv_path)
    summaries = []
    for var in variants:
        summary = apply_variant_and_save(df_raw, var, out_base=out_base)
        summaries.append(summary)
    return summaries
