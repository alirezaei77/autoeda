"""
Command-line interface for AutoEDA.

Usage examples
--------------
$ python -m autoeda.cli --input data/iris.csv --target species --output reports/iris
$ python -m autoeda.cli --builtin iris --output reports/iris
$ python -m autoeda.cli --compare --builtin-all --output reports/all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn import datasets


BUILTIN_LOADERS = {
    "iris": lambda: datasets.load_iris(as_frame=True).frame.rename(
        columns={"target": "target"}
    ),
    "wine": lambda: datasets.load_wine(as_frame=True).frame.rename(
        columns={"target": "target"}
    ),
    "california": lambda: datasets.fetch_california_housing(as_frame=True).frame.rename(
        columns={"MedHouseVal": "target"}
    ),
}


def _load_titanic() -> pd.DataFrame:
    """Load Titanic from seaborn (cached local copy)."""
    import seaborn as sns

    df = sns.load_dataset("titanic")
    return df


def _load_builtin(name: str) -> pd.DataFrame:
    name = name.lower()
    if name == "titanic":
        return _load_titanic()
    if name in BUILTIN_LOADERS:
        return BUILTIN_LOADERS[name]()
    raise ValueError(f"Unknown builtin dataset: {name}")


def _run_single(args: argparse.Namespace) -> int:
    from .analyzer import AutoEDA

    if args.builtin:
        df = _load_builtin(args.builtin)
        name = args.builtin
    else:
        if not args.input:
            print("error: --input or --builtin required", file=sys.stderr)
            return 2
        df = pd.read_csv(args.input)
        name = Path(args.input).stem

    eda = AutoEDA(df, dataset_name=name, target=args.target)
    report = eda.run(args.output)
    print(
        f"[AutoEDA] {name}: {report.shape[0]}x{report.shape[1]} | "
        f"problem={report.problem_info['problem_type']} | "
        f"coverage={report.coverage_pct}% | report -> {args.output}/report.html"
    )
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    from .compare import compare_datasets

    names = ["iris", "wine", "titanic", "california"]
    datasets = {n: _load_builtin(n) for n in names}
    targets = {"iris": "target", "wine": "target", "titanic": "survived",
               "california": "target"}
    cmp_df = compare_datasets(datasets, args.output, targets=targets)
    print(cmp_df.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoeda",
        description="Automated Exploratory Data Analysis toolkit.",
    )
    parser.add_argument("--input", type=str, help="Path to CSV file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--target", type=str, default=None,
                        help="Target column name (optional).")
    parser.add_argument("--builtin", type=str, default=None,
                        choices=["iris", "wine", "titanic", "california"],
                        help="Run on a builtin dataset.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all builtin datasets side-by-side.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.compare:
        return _run_compare(args)
    return _run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())