"""Test direct forecasting function."""

import numpy as np
import pytest

from aeon.classification import DummyClassifier
from aeon.forecasting import ETSForecaster, direct_forecasting


def test_direct_forecasting():
    """Test for the direct forecasting function."""
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(
        TypeError, match="Passed an object in the forecaster_class parameter"
    ):
        direct_forecasting(ETSForecaster(), y, 10)
    with pytest.raises(
        TypeError, match="Passed a class in the forecaster_class parameter"
    ):
        direct_forecasting(DummyClassifier, y, 10)
    pred = direct_forecasting(ETSForecaster, y, steps_ahead=5)
    assert len(pred) == 5
    assert np.allclose(
        pred, np.array([4.486784, 4.486784, 4.486784, 4.486784, 4.486784])
    )
