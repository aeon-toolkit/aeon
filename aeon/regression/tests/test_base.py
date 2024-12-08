"""Unit tests for regression base class functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from aeon.datasets import load_covid_3month
from aeon.regression._dummy import DummyRegressor
from aeon.regression.base import BaseRegressor
from aeon.testing.testing_data import (
    EQUAL_LENGTH_UNIVARIATE_REGRESSION,
    UNEQUAL_LENGTH_UNIVARIATE_REGRESSION,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES


class _TestRegressor(BaseRegressor):
    """Dummy regressor for testing base class fit/predict."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.random.random(size=(len(X)))


class _DummyHandlesAllInput(BaseRegressor):
    """Dummy regressor for testing base class fit/predict."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.random.random(size=(len(X)))


class _TestHandlesAllInput(BaseRegressor):
    """Dummy regressor for testing base class fit/predict/predict_proba."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.random.random(size=(len(X)))


multivariate_message = r"multivariate series"
missing_message = r"missing values"
unequal_message = r"unequal length series"
incorrect_X_data_structure = r"must be a np.array or a pd.Series"
incorrect_y_data_structure = r"must be 1-dimensional"


def _assert_fit_predict(dummy, X, y):
    result = dummy.fit(X, y)
    # Fit returns self
    assert result is dummy
    preds = dummy.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 10


def test__check_y():
    """Test private method _check_y."""
    # Correct outcomes
    reg = _TestRegressor()
    y = np.random.random(size=(100))
    reg._check_y(y, 100)
    assert isinstance(y, np.ndarray)
    y = pd.Series(y)
    y = reg._check_y(y, 100)
    assert isinstance(y, np.ndarray)
    # Test error raising
    # y wrong length
    with pytest.raises(ValueError, match=r"Mismatch in number of cases"):
        reg._check_y(y, 99)
    # y invalid type
    y = ["This", "is", "tested", "lots"]
    with pytest.raises(TypeError, match=r"np.array or a pd.Series"):
        reg._check_y(y, 4)
    y = np.ndarray([1, 2, 1, 2, 1, 2])
    with pytest.raises(TypeError, match=r"y must be 1-dimensional"):
        reg._check_y(y, 6)
    y = np.array(["1.1", "2.2", "3.3", "4.4", "5.5"])
    with pytest.raises(ValueError, match=r"contains strings, cannot fit a regressor"):
        reg._check_y(y, 5)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_unequal_length_input(data):
    """Test with unequal length failures and passes."""
    if data in UNEQUAL_LENGTH_UNIVARIATE_REGRESSION.keys():
        dummy = _TestRegressor()
        X = UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[data]["train"][0]
        y = np.random.random(size=10)
        with pytest.raises(ValueError, match=r"has unequal length series, but"):
            dummy.fit(X, y)
        dummy = _TestHandlesAllInput()
        _assert_fit_predict(dummy, X, y)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_equal_length_input(data):
    """Test with unequal length failures and passes."""
    dummy = _TestRegressor()
    X = EQUAL_LENGTH_UNIVARIATE_REGRESSION[data]["train"][0]
    y = np.random.random(size=10)
    _assert_fit_predict(dummy, X, y)
    dummy = _TestHandlesAllInput()
    _assert_fit_predict(dummy, X, y)


def test_base_regressor_fit():
    """Test function for the BaseRegressor class fit.

    Test fit. It should:
    1. Work with 2D, 3D and list of np.ndarray for X and np.ndarray for y.
    2. Calculate the number of classes and record the fit time.
    3. have self.n_jobs set or throw  an exception if the classifier can
    multithread.
    4. Set the class dictionary correctly.
    5. Set is_fitted after a call to _fit.
    6. Return self.
    """
    dummy = _TestRegressor()
    cases = 5
    length = 10
    test_X1 = np.random.uniform(-1, 1, size=(cases, length))
    test_X2 = np.random.uniform(-1, 1, size=(cases, 2, length))
    test_y1 = np.random.random(cases)
    result = dummy.fit(test_X1, test_y1)
    assert result is dummy
    with pytest.raises(ValueError, match=multivariate_message):
        result = dummy.fit(test_X2, test_y1)
    assert result is dummy
    # Raise a specific error if y is in a 2D matrix (1,cases)
    np.array([test_y1])
    # What if y is in a 2D matrix (cases,1)?
    test_y2 = np.array([test_y1]).transpose()
    with pytest.raises(TypeError, match=incorrect_y_data_structure):
        dummy.fit(test_X1, test_y2)


def test_score():
    """Test score with continuous and binary regression."""
    dummy = DummyRegressor()
    x_train, y_train = load_covid_3month(split="train")
    x_test, y_test = load_covid_3month(split="test")
    dummy.fit(x_train, y_train)
    r = dummy.score(x_test, y_test)
    np.testing.assert_almost_equal(r, -0.004303695576216793, decimal=6)
    with pytest.raises(ValueError):
        dummy.score(x_test, y_test, metric="r3")
    r = dummy.score(x_test, y_test, metric=r2_score)  # Use callable
    np.testing.assert_almost_equal(r, -0.004303695576216793, decimal=6)
