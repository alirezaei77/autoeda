import numpy as np
import pandas as pd
import pytest

from autoeda import stats_utils as S


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 100],
            "b": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "c": ["x", "y", "x", "y", "x", "y"],
        }
    )


def test_numeric_summary(df):
    s = S.numeric_summary(df)
    assert "a" in s.index
    assert "missing" in s.columns


def test_missing_summary(df):
    s = S.missing_summary(df)
    assert s.loc["b", "missing_count"] == 1


def test_outliers_iqr_flags_extreme():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    mask, low, up = S.outliers_iqr(s)
    assert mask[-1] is np.True_ or mask[-1] == True


def test_outliers_zscore():
    s = pd.Series(np.random.RandomState(0).normal(size=1000))
    mask = S.outliers_zscore(s, threshold=3)
    assert mask.sum() < 20


def test_correlation_matrices():
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 2})
    mats = S.correlation_matrices(df)
    assert "pearson" in mats
    assert mats["pearson"].loc["a", "b"] == pytest.approx(1.0)


def test_normality_tests():
    df = pd.DataFrame({"a": np.random.RandomState(1).normal(size=200)})
    res = S.normality_tests(df)
    assert "shapiro_p" in res.columns