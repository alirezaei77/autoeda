"""
Plot generation utilities for AutoEDA.

All plots are saved as PNG files inside an output directory and referenced
from the generated HTML report.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir: Path, name: str) -> str:
    """Save a matplotlib figure and return the relative filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_missing_matrix(df: pd.DataFrame, out_dir: Path) -> Optional[str]:
    if df.isna().sum().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    msno.matrix(df, ax=ax, sparkline=False)
    return _save(fig, out_dir, "missing_matrix.png")


def plot_missing_bar(df: pd.DataFrame, out_dir: Path) -> Optional[str]:
    if df.isna().sum().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    msno.bar(df, ax=ax)
    return _save(fig, out_dir, "missing_bar.png")


def plot_histograms(df: pd.DataFrame, out_dir: Path, max_cols: int = 20) -> Optional[str]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return None
    cols = list(num_df.columns)[:max_cols]
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        sns.histplot(num_df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis("off")
    return _save(fig, out_dir, "histograms.png")


def plot_boxplots(df: pd.DataFrame, out_dir: Path, max_cols: int = 20) -> Optional[str]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return None
    cols = list(num_df.columns)[:max_cols]
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        sns.boxplot(x=num_df[col].dropna(), ax=ax, color="salmon")
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis("off")
    return _save(fig, out_dir, "boxplots.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path, method: str = "pearson") -> Optional[str]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(max(6, num_df.shape[1] * 0.7),
                                    max(5, num_df.shape[1] * 0.6)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title(f"{method.capitalize()} correlation")
    return _save(fig, out_dir, f"corr_{method}.png")


def plot_categorical_bars(df: pd.DataFrame, out_dir: Path, top_k: int = 10) -> Optional[str]:
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if cat_df.empty:
        return None
    cols = list(cat_df.columns)[:9]
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        vc = cat_df[col].value_counts().head(top_k)
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color="teal")
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis("off")
    return _save(fig, out_dir, "categorical_bars.png")


def plot_target_distribution(
    df: pd.DataFrame, target: str, out_dir: Path
) -> Optional[str]:
    if target not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    s = df[target]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
        sns.histplot(s.dropna(), kde=True, ax=ax, color="purple")
    else:
        vc = s.value_counts()
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, color="purple")
    ax.set_title(f"Target distribution: {target}")
    return _save(fig, out_dir, "target_distribution.png")