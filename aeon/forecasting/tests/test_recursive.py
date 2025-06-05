"""Test recursive function."""

import numpy as np
import pytest

from aeon.classification import DummyClassifier
from aeon.forecasting import RegressionForecaster, recursive_forecasting


def test_recursive_forecasting():
    """Test for the direct forecasting function."""
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(TypeError, match="Passed a class in the forecaster parameter"):
        recursive_forecasting(RegressionForecaster, y, 5)
    with pytest.raises(
        TypeError, match="Passed an object that does not inherit from BaseForecaster"
    ):
        recursive_forecasting(DummyClassifier(), y, 5)
    pred = recursive_forecasting(RegressionForecaster(window=3), y, steps_ahead=5)
    assert len(pred) == 5
    assert np.allclose(pred, np.array([11.0, 12.0, 13.0, 14.0, 15.0]))
