from pathlib import Path

import pandas as pd

from autoeda import AutoEDA


def _toy_df():
    return pd.DataFrame(
        {
            "age": [22, 25, 47, 52, 46, 56, 23, 30, 40, 60],
            "fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 15.5, 30.0],
            "sex": ["m", "f", "f", "f", "m", "m", "m", "f", "f", "m"],
            "survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        }
    )


def test_run_creates_report(tmp_path: Path):
    eda = AutoEDA(_toy_df(), dataset_name="toy", target="survived")
    report = eda.run(tmp_path, make_plots=True, export_html=True, export_json=True)

    assert report.shape == (10, 4)
    assert report.problem_info["problem_type"] == "binary classification"
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.json").exists()
    assert report.coverage_pct > 0


def test_run_on_iris():
    from sklearn.datasets import load_iris

    df = load_iris(as_frame=True).frame
    eda = AutoEDA(df, dataset_name="iris", target="target")
    report = eda.run("outputs/_test_iris", make_plots=False, export_html=False)
    assert report.coverage_pct >= 90.0