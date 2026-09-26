"""
Run AutoEDA on 4 reference datasets and compare them.
"""

from pathlib import Path

import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris, load_wine

from autoeda import AutoEDA, compare_datasets

OUT = Path("outputs/examples")


def load_all() -> dict:
    iris = load_iris(as_frame=True).frame
    wine = load_wine(as_frame=True).frame
    titanic = sns.load_dataset("titanic")
    cal = fetch_california_housing(as_frame=True).frame.rename(
        columns={"MedHouseVal": "target"}
    )
    return {"iris": iris, "wine": wine, "titanic": titanic, "california": cal}


def main() -> None:
    datasets = load_all()
    targets = {
        "iris": "target",
        "wine": "target",
        "titanic": "survived",
        "california": "target",
    }

    for name, df in datasets.items():
        print(f"\n=== {name}: {df.shape} ===")
        eda = AutoEDA(df, dataset_name=name, target=targets[name])
        report = eda.run(OUT / name)
        print(
            f"  problem={report.problem_info['problem_type']}, "
            f"coverage={report.coverage_pct}%"
        )

    print("\n=== Comparison ===")
    comparison = compare_datasets(datasets, OUT / "_comparison", targets=targets)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()