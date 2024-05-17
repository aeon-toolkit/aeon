"""Test for Hampel filter."""

import pandas as pd

from aeon.anomaly_detection._hampel import HampelFilter


def test_predict_outliers():
    """Test internal predict outliers function."""
    hf = HampelFilter(window_length=4)
    x = pd.Series([1, 2, 3, 100, 3, 2, 1, 5])
    x2 = hf._predict_outliers(x)
    assert isinstance(x2, pd.Series)
    assert x2.iloc[3]
    assert not x2.iloc[0] and not x2.iloc[6]
