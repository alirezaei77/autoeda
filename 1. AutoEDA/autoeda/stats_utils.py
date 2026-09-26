"""
Statistical utilities used across the AutoEDA pipeline.

Implements:
- Descriptive statistics (numeric + categorical)
- Missing value summary
- IQR and Z-score based outlier detection
- Normality tests (Shapiro, Anderson-Darling)
- Correlation matrices (Pearson, Spearman)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return extended descriptive statistics for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    desc = num_df.describe().T
    desc["median"] = num_df.median()
    desc["mode"] = num_df.mode().iloc[0] if not num_df.mode().empty else np.nan
    desc["skew"] = num_df.skew()
    desc["kurtosis"] = num_df.kurtosis()
    desc["variance"] = num_df.var()
    desc["cv"] = desc["std"] / desc["mean"].replace(0, np.nan)
    desc["missing"] = num_df.isna().sum()
    desc["missing_pct"] = (desc["missing"] / len(num_df)) * 100
    return desc.round(4)


def categorical_summary(df: pd.DataFrame, top_k: int = 10) -> Dict[str, pd.DataFrame]:
    """Return per-column frequency tables for categorical features."""
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    result: Dict[str, pd.DataFrame] = {}
    for col in cat_df.columns:
        counts = cat_df[col].value_counts(dropna=False).head(top_k)
        freq = (counts / len(cat_df) * 100).round(2)
        result[col] = pd.DataFrame({"count": counts, "percent": freq})
    return result


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of missing values per column."""
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    out = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    out["dtype"] = df.dtypes.astype(str)
    out = out.sort_values("missing_pct", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def outliers_iqr(series: pd.Series, k: float = 1.5) -> Tuple[np.ndarray, float, float]:
    """
    Detect outliers using the IQR rule.

    Returns
    -------
    mask : np.ndarray[bool]
        Boolean mask of outliers.
    lower : float
        Lower bound (Q1 - k*IQR).
    upper : float
        Upper bound (Q3 + k*IQR).
    """
    s = series.dropna()
    if s.empty:
        return np.zeros(len(series), dtype=bool), np.nan, np.nan

    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return mask.fillna(False).to_numpy(), float(lower), float(upper)


def outliers_zscore(series: pd.Series, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score threshold (default 3 sigma)."""
    s = series.dropna()
    if s.empty or s.std() == 0:
        return np.zeros(len(series), dtype=bool)
    z = np.abs(stats.zscore(s, nan_policy="omit"))
    mask = pd.Series(False, index=series.index)
    mask.loc[s.index] = z > threshold
    return mask.to_numpy()


def outlier_report(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """Return a per-numeric-column outlier summary."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    rows: List[Dict] = []
    for col in num_cols:
        if method == "iqr":
            mask, low, up = outliers_iqr(df[col])
            rows.append(
                {
                    "column": col,
                    "method": "IQR",
                    "n_outliers": int(mask.sum()),
                    "pct_outliers": round(mask.sum() / len(df) * 100, 2),
                    "lower_bound": low,
                    "upper_bound": up,
                }
            )
        else:
            mask = outliers_zscore(df[col])
            rows.append(
                {
                    "column": col,
                    "method": "Z-score",
                    "n_outliers": int(mask.sum()),
                    "pct_outliers": round(mask.sum() / len(df) * 100, 2),
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def correlation_matrices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return Pearson and Spearman correlation matrices for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    out: Dict[str, pd.DataFrame] = {}
    if num_df.shape[1] >= 2:
        out["pearson"] = num_df.corr(method="pearson").round(4)
        out["spearman"] = num_df.corr(method="spearman").round(4)
    return out


# ---------------------------------------------------------------------------
# Normality tests
# ---------------------------------------------------------------------------

def normality_tests(df: pd.DataFrame, sample: int = 5000) -> pd.DataFrame:
    """Run Shapiro-Wilk and Anderson-Darling tests per numeric column."""
    rows: List[Dict] = []
    num_df = df.select_dtypes(include=[np.number])
    for col in num_df.columns:
        s = num_df[col].dropna()
        if s.empty or s.nunique() < 3:
            continue
        if len(s) > sample:
            s = s.sample(sample, random_state=42)

        try:
            shapiro_stat, shapiro_p = stats.shapiro(s)
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan

        try:
            ad = stats.anderson(s, dist="norm")
            ad_stat = ad.statistic
            ad_crit = ad.critical_values[2]  # 5% level
            is_normal_ad = ad_stat < ad_crit
        except Exception:
            ad_stat, is_normal_ad = np.nan, None

        rows.append(
            {
                "column": col,
                "shapiro_stat": round(shapiro_stat, 4) if not np.isnan(shapiro_stat) else np.nan,
                "shapiro_p": round(shapiro_p, 6) if not np.isnan(shapiro_p) else np.nan,
                "is_normal_shapiro": (shapiro_p > 0.05) if not np.isnan(shapiro_p) else None,
                "anderson_stat": round(ad_stat, 4) if not np.isnan(ad_stat) else np.nan,
                "is_normal_anderson": is_normal_ad,
            }
        )
    return pd.DataFrame(rows)