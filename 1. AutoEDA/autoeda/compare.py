"""
Compare multiple datasets side-by-side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .analyzer import AutoEDA


def compare_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_dir: str | Path,
    targets: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Run AutoEDA on several datasets and return a comparison table.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of dataset name -> DataFrame.
    output_dir : str | Path
        Where per-dataset reports are written.
    targets : dict[str, str], optional
        Optional mapping of dataset name -> target column.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    targets = targets or {}

    rows: List[dict] = []
    for name, df in datasets.items():
        target = targets.get(name)
        eda = AutoEDA(df, dataset_name=name, target=target)
        report = eda.run(out_root / name)

        rows.append(
            {
                "dataset": name,
                "n_rows": report.shape[0],
                "n_cols": report.shape[1],
                "numeric_cols": int(len(report.numeric_summary)),
                "categorical_cols": int(len(report.categorical_summary)),
                "missing_pct_total": round(
                    float(report.missing_summary["missing_pct"].mean()), 2
                )
                if not report.missing_summary.empty
                else 0.0,
                "total_outliers_iqr": int(report.outlier_iqr["n_outliers"].sum())
                if not report.outlier_iqr.empty
                else 0,
                "problem_type": report.problem_info["problem_type"],
                "coverage_pct": report.coverage_pct,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    comparison.to_csv(out_root / "comparison.csv", index=False)
    comparison.to_html(out_root / "comparison.html", index=False)
    return comparison