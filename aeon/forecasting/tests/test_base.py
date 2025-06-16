"""Test base forecaster."""

import numpy as np
import pandas as pd
import pytest

from aeon.forecasting import NaiveForecaster, RegressionForecaster


def test_base_forecaster():
    """Test base forecaster functionality."""
    f = NaiveForecaster()
    y = np.random.rand(50)
    f.fit(y)
    p1 = f.predict()
    assert p1 == y[-1]
    p2 = f.forecast(y)
    p3 = f._forecast(y)
    assert p2 == p1
    assert p3 == p2
    with pytest.raises(ValueError, match="Exogenous variables passed"):
        f.fit(y, exog=y)


def test_convert_y():
    """Test y conversion in forecasting base."""
    f = NaiveForecaster()
    y = np.random.rand(50)
    with pytest.raises(ValueError, match="Input axis should be 0 or 1"):
        f._convert_y(y, axis=2)
    y2 = f._convert_y(pd.Series(y), axis=0)
    assert isinstance(y2, np.ndarray)
    y = np.random.random((100, 2))
    y2 = f._convert_y(y, axis=0)
    assert y2.shape == (2, 100)
    f.set_tags(**{"y_inner_type": "pd.DataFrame"})
    y2 = f._convert_y(y, axis=0)
    assert isinstance(y2, pd.DataFrame)
    f.set_tags(**{"y_inner_type": "pd.Series"})
    with pytest.raises(ValueError, match="Unsupported inner type"):
        f._convert_y(y, axis=1)


def test_direct_forecast():
    """Test direct forecasting."""
    y = np.random.rand(50)
    f = RegressionForecaster(window=10)
    # Direct should be the same as setting horizon manually.
    preds = f.direct_forecast(y, prediction_horizon=10)
    assert isinstance(preds, np.ndarray) and len(preds) == 10
    for i in range(1, 11):
        f = RegressionForecaster(window=10, horizon=i)
        p = f.forecast(y)
        assert p == preds[i - 1]
