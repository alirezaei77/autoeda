import numpy as np
import pandas as pd

from autoeda.problem_detector import detect_problem_type


def test_binary_classification():
    df = pd.DataFrame({"x": np.arange(100), "y": [0, 1] * 50})
    info = detect_problem_type(df, target="y")
    assert info["problem_type"] == "binary classification"


def test_multiclass_classification():
    df = pd.DataFrame({"x": np.arange(90), "y": [0, 1, 2] * 30})
    info = detect_problem_type(df, target="y")
    assert info["problem_type"] == "multiclass classification"
    assert info["n_classes"] == 3


def test_regression():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x": np.arange(500), "y": rng.normal(size=500)})
    info = detect_problem_type(df, target="y")
    assert info["problem_type"] == "regression"