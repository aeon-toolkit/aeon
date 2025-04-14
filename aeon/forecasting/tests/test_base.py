"""Test base forecaster."""

import numpy as np
import pandas as pd
import pytest

from aeon.forecasting import DummyForecaster


def test_base_forecaster():
    """Test base forecaster functionality."""
    f = DummyForecaster()
    y = np.random.rand(50)
    f.fit(y)
    p1 = f.predict()
    assert p1 == y[-1]
    p2 = f.forecast(y)
    p3 = f._forecast(y)
    assert p2 == p1
    assert p3 == p2
    with pytest.raises(
        NotImplementedError, match="Exogenous variables not yet " "supported"
    ):
        f.fit(y, exog=y)


def test_convert_y():
    """Test y conversion in forecasting base."""
    f = DummyForecaster()
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
