"""
Core AutoEDA class: orchestrates the full analysis pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import plots as P
from . import stats_utils as S
from .problem_detector import detect_problem_type


@dataclass
class AutoEDAReport:
    """Container holding the analysis artifacts."""

    dataset_name: str
    shape: tuple
    numeric_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    categorical_summary: Dict[str, pd.DataFrame] = field(default_factory=dict)
    missing_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    outlier_iqr: pd.DataFrame = field(default_factory=pd.DataFrame)
    outlier_z: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlations: Dict[str, pd.DataFrame] = field(default_factory=dict)
    normality: pd.DataFrame = field(default_factory=pd.DataFrame)
    problem_info: dict = field(default_factory=dict)
    plots: List[str] = field(default_factory=list)
    coverage_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "shape": list(self.shape),
            "coverage_pct": self.coverage_pct,
            "problem_info": self.problem_info,
            "plots": self.plots,
        }


class AutoEDA:
    """
    Automated Exploratory Data Analysis.

    Example
    -------
    >>> eda = AutoEDA(df, dataset_name="iris", target="species")
    >>> report = eda.run(output_dir="reports/iris")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        target: Optional[str] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df is empty")
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.target = target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        output_dir: str | Path,
        make_plots: bool = True,
        export_html: bool = True,
        export_json: bool = True,
    ) -> AutoEDAReport:
        """Run the full EDA pipeline and persist outputs."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        report = AutoEDAReport(
            dataset_name=self.dataset_name,
            shape=self.df.shape,
        )

        # --- statistics -------------------------------------------------
        report.numeric_summary = S.numeric_summary(self.df)
        report.categorical_summary = S.categorical_summary(self.df)
        report.missing_summary = S.missing_summary(self.df)
        report.outlier_iqr = S.outlier_report(self.df, method="iqr")
        report.outlier_z = S.outlier_report(self.df, method="zscore")
        report.correlations = S.correlation_matrices(self.df)
        report.normality = S.normality_tests(self.df)
        report.problem_info = detect_problem_type(self.df, self.target)

        # --- coverage ---------------------------------------------------
        total_cols = self.df.shape[1]
        covered = 0
        covered += len(report.numeric_summary)
        covered += len(report.categorical_summary)
        covered += len(report.normality)
        report.coverage_pct = round(min(covered, total_cols) / total_cols * 100, 2)

        # --- plots ------------------------------------------------------
        if make_plots:
            plot_dir = out_dir / "plots"
            report.plots = self._generate_plots(plot_dir)

        # --- exports ----------------------------------------------------
        if export_json:
            self._export_json(report, out_dir / "report.json")

        if export_html:
            from .report import build_html_report

            build_html_report(report, out_dir / "report.html", out_dir)

        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_plots(self, plot_dir: Path) -> List[str]:
        names: List[str] = []
        for fn in (
            lambda: P.plot_missing_matrix(self.df, plot_dir),
            lambda: P.plot_missing_bar(self.df, plot_dir),
            lambda: P.plot_histograms(self.df, plot_dir),
            lambda: P.plot_boxplots(self.df, plot_dir),
            lambda: P.plot_categorical_bars(self.df, plot_dir),
            lambda: P.plot_correlation_heatmap(self.df, plot_dir, "pearson"),
            lambda: P.plot_correlation_heatmap(self.df, plot_dir, "spearman"),
            lambda: P.plot_target_distribution(self.df, self.target, plot_dir)
            if self.target
            else None,
        ):
            try:
                name = fn()
                if name:
                    names.append(name)
            except Exception as exc:  # pragma: no cover
                print(f"[AutoEDA] plot error: {exc}")
        return names

    @staticmethod
    def _export_json(report: AutoEDAReport, path: Path) -> None:
        def _default(o):
            if isinstance(o, pd.DataFrame):
                return o.to_dict()
            return str(o)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=_default)