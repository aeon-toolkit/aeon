"""Test base forecaster."""

import numpy as np
import pandas as pd
import pytest

from aeon.forecasting import NaiveForecaster, RegressionForecaster
from aeon.forecasting.base import BaseForecaster


def test_base_forecaster():
    """Test base forecaster functionality."""
    f = NaiveForecaster()
    y = np.random.rand(50)
    f.fit(y)
    p1 = f.predict(y)
    assert p1 == y[-1]
    p2 = f.forecast(y)
    p3 = f._forecast(y, None)
    assert p2 == p1
    assert p3 == p2
    with pytest.raises(ValueError, match="Exogenous variables passed"):
        f.forecast(y, exog=y)


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
    y2 = f._convert_y(y, axis=1)
    assert isinstance(y2, pd.DataFrame)
    f.set_tags(**{"y_inner_type": "pd.Series"})
    with pytest.raises(ValueError, match="Unsupported inner type"):
        f._convert_y(y, axis=1)
    with pytest.raises(ValueError, match="must be greater than or equal to 1"):
        f.direct_forecast(y, prediction_horizon=0)


def test_direct_forecast():
    """Test direct forecasting."""
    y = np.random.rand(50)
    f = RegressionForecaster(window=10)
    # Direct should be the same as setting horizon manually.
    preds = f.direct_forecast(y, prediction_horizon=10)
    assert isinstance(preds, np.ndarray) and len(preds) == 10
    for i in range(0, 10):
        f = RegressionForecaster(window=10, horizon=i + 1)
        p = f.forecast(y)
        assert p == preds[i]


def test_iterative_forecast():
    """Test terativeforecasting."""
    y = np.random.rand(50)
    f = RegressionForecaster(window=4)
    preds = f.iterative_forecast(y, prediction_horizon=10)
    assert isinstance(preds, np.ndarray) and len(preds) == 10
    f.fit(y)
    for i in range(0, 10):
        p = f.predict(y)
        assert p == preds[i]
        y = np.append(y, p)


def test_output_equivalence():
    """Test output same for one ahead forecast."""
    y = np.random.rand(50)
    f = RegressionForecaster(window=4)
    p1 = f.forecast(y)
    p2 = f.fit(y).predict(y)
    p3 = f.iterative_forecast(y, 1)
    p4 = f.direct_forecast(y, 1)
    assert np.allclose(p1, p2, p3[0], p4[0])


def test_direct_forecast_with_exog():
    """Test direct forecasting with exogenous variables."""
    y = np.arange(50)
    exog = np.arange(50) * 2
    f = RegressionForecaster(window=10)

    preds = f.direct_forecast(y, prediction_horizon=10, exog=exog)
    assert isinstance(preds, np.ndarray) and len(preds) == 10

    # Check that predictions are different from when no exog is used
    preds_no_exog = f.direct_forecast(y, prediction_horizon=10)
    assert not np.array_equal(preds, preds_no_exog)


def test_fit_is_empty():
    """Test empty fit."""

    class _EmptyFit(BaseForecaster):
        _tags = {"fit_is_empty": True}

        def _fit(self, y):
            return self

        def _predict(self, y):
            return 0

    dummy = _EmptyFit(horizon=1, axis=1)
    y = np.arange(50)
    dummy.fit(y)
    assert dummy.is_fitted
