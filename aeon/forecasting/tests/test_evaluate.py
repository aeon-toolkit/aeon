"""Test the forecast evaluator."""

import numpy as np
import pytest

from aeon.classification import DummyClassifier
from aeon.forecasting import ETSForecaster, RegressionForecaster, evaluate_forecaster


def test_evaluate_forecaster():
    """Test for the direct forecasting function."""
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_test = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
    with pytest.raises(TypeError, match="Passed a class in the forecaster parameter"):
        evaluate_forecaster(RegressionForecaster, y_train, y_test)
    with pytest.raises(
        TypeError, match="Passed an object that does not inherit from BaseForecaster"
    ):
        evaluate_forecaster(DummyClassifier(), y_train, y_test)
    pred = evaluate_forecaster(RegressionForecaster(window=3), y_train, y_test)
    assert len(pred) == 5
    assert np.allclose(pred, np.array([11.0, 12.0, 13.0, 14.0, 15.0]))
    pred = evaluate_forecaster(ETSForecaster(), y_train, y_test)
    assert len(pred) == 5
    assert np.allclose(
        pred, np.array([4.4867844, 5.13810596, 5.82429536, 6.54186583, 7.28767925])
    )
