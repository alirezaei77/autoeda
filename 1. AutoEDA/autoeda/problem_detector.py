"""
Heuristic detection of the machine-learning problem type of a dataset:
- Binary classification
- Multiclass classification
- Regression
- Clustering / unsupervised
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def detect_problem_type(
    df: pd.DataFrame, target: Optional[str] = None
) -> dict:
    """
    Heuristically detect the ML problem type.

    Parameters
    ----------
    df : pd.DataFrame
    target : str, optional
        Target column name. If None, attempts to guess the last column or
        falls back to unsupervised.

    Returns
    -------
    dict with keys: problem_type, target, n_classes, rationale
    """
    info = {
        "problem_type": "unknown",
        "target": target,
        "n_classes": None,
        "rationale": "",
    }

    if target is None:
        # Heuristic: last column is often the target in sklearn datasets
        target = df.columns[-1]
        info["target"] = target
        info["rationale"] = "No target provided; using last column as target."

    if target not in df.columns:
        info["problem_type"] = "unsupervised (clustering / EDA)"
        info["target"] = None
        info["rationale"] = f"Column '{target}' not found."
        return info

    y = df[target]
    n_unique = y.nunique(dropna=True)
    dtype = y.dtype

    if pd.api.types.is_float_dtype(dtype) and n_unique > 20:
        # Continuous numeric target
        ratio_unique = n_unique / len(y)
        if ratio_unique > 0.05:
            info["problem_type"] = "regression"
            info["rationale"] = (
                f"Target '{target}' is continuous with {n_unique} unique values."
            )
            return info

    if n_unique == 2:
        info["problem_type"] = "binary classification"
        info["n_classes"] = 2
    elif 2 < n_unique <= 50:
        info["problem_type"] = "multiclass classification"
        info["n_classes"] = int(n_unique)
    elif n_unique > 50:
        info["problem_type"] = "regression (or high-cardinality target)"
        info["n_classes"] = int(n_unique)
    else:
        info["problem_type"] = "unsupervised (clustering / EDA)"

    info["rationale"] += (
        f" Target dtype={dtype}, unique values={n_unique}."
    )
    return info