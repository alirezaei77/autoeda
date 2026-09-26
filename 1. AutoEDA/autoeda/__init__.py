"""
AutoEDA - Automated Exploratory Data Analysis toolkit.

A lightweight, dependency-friendly toolkit that generates a full EDA report
(HTML + plots + summary tables) for any tabular dataset.
"""

from .analyzer import AutoEDA
from .problem_detector import detect_problem_type
from .compare import compare_datasets

__version__ = "0.1.0"
__all__ = ["AutoEDA", "detect_problem_type", "compare_datasets"]