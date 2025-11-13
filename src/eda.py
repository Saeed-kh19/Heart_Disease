"""
eda.py
Functions to create and save common EDA figures:
- histograms
- boxplots
- qqplots
- correlation heatmap
- countplots
- optional pairplot (subset)
Each plotting function saves a PNG into the provided outdir and returns the filepath.
"""

import os
from typing import List, Sequence, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

sns.set(style="whitegrid", context="talk")

def ensure_dir(path: str):
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)

def _save_fig(fig, filepath: str):
    """Tighten, save and close a Matplotlib figure to avoid memory leaks."""
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath

def plot_histograms(df: pd.DataFrame, cols: Sequence[str], outdir: str) -> List[str]:
    """Create and save one histogram per column in cols. Returns list of filepaths."""
    ensure_dir(outdir)
    saved = []
    for col in cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2b8cbe")
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        path = os.path.join(outdir, f"hist_{col}.png")
        saved.append(_save_fig(fig, path))
    return saved

def plot_boxplots(df: pd.DataFrame, cols: Sequence[str], outdir: str) -> List[str]:
    """Create and save boxplot per column in cols. Returns list of filepaths."""
    ensure_dir(outdir)
    saved = []
    for col in cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[col], color="#f03b20", ax=ax)
        ax.set_title(f"Boxplot of {col}")
        ax.set_xlabel(col)
        path = os.path.join(outdir, f"box_{col}.png")
        saved.append(_save_fig(fig, path))
    return saved

def plot_qq(df: pd.DataFrame, col: str, outdir: str) -> str:
    """Create and save a Q-Q plot for a single column and return filepath."""
    ensure_dir(outdir)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    stats.probplot(df[col].dropna(), dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot of {col}")
    path = os.path.join(outdir, f"qq_{col}.png")
    return _save_fig(fig, path)

def plot_corr_heatmap(df: pd.DataFrame, cols: Sequence[str], outdir: str, annot: bool = True) -> str:
    """Compute correlation matrix over cols, plot heatmap, save and return filepath."""
    ensure_dir(outdir)
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    path = os.path.join(outdir, "corr_heatmap.png")
    return _save_fig(fig, path)

def plot_countplots(df: pd.DataFrame, cols: Sequence[str], outdir: str) -> List[str]:
    """Create vertical countplots for given categorical/binary cols and save them."""
    ensure_dir(outdir)
    saved = []
    for col in cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df[col], order=sorted(df[col].dropna().unique()), palette="muted", ax=ax)
        ax.set_title(f"Countplot of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        path = os.path.join(outdir, f"count_{col}.png")
        saved.append(_save_fig(fig, path))
    return saved

def plot_pairplot(df: pd.DataFrame, cols: Sequence[str], outdir: str, hue: Optional[str] = None) -> str:
    """
    Create and save a seaborn pairplot for a subset of columns.
    Use sparingly for a small number of features to avoid performance issues.
    """
    ensure_dir(outdir)
    pp = sns.pairplot(df[list(cols) + ([hue] if hue else [])].dropna(), hue=hue, diag_kind="kde", palette="crest")
    path = os.path.join(outdir, "pairplot_subset.png")
    # Seaborn's PairGrid uses plt.savefig; avoid closing the figure that PairGrid manages directly
    pp.savefig(path)
    plt.close("all")
    return path
