# -*- coding: utf-8 -*-
"""Unit tests for regression base class functionality."""
import numpy as np
import pytest

from aeon.datasets import load_unit_test
from aeon.regression._dummy import DummyRegressor
from aeon.regression.base import BaseRegressor


class _DummyRegressor(BaseRegressor):
    """Dummy regressor for testing base class fit/predict."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self


class _DummyHandlesAllInput(BaseRegressor):
    """Dummy regress for testing base class fit/predict."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_mtype": ["np-list", "numpy3D"],
    }

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self


multivariate_message = r"multivariate series"
missing_message = r"missing values"
unequal_message = r"unequal length series"
incorrect_X_data_structure = r"must be a np.array or a pd.Series"
incorrect_y_data_structure = r"must be 1-dimensional"


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
    dummy = _DummyRegressor()
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
    test_y2 = np.array([test_y1])
    # What if y is in a 2D matrix (cases,1)?
    test_y2 = np.array([test_y1]).transpose()
    with pytest.raises(ValueError, match=incorrect_y_data_structure):
        result = dummy.fit(test_X1, test_y2)


@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("unequal", [True, False])
def test_check_capabilities(missing, multivariate, unequal):
    """Test the checking of capabilities."""
    handles_none = _DummyRegressor()

    # checks that errors are raised
    if missing:
        with pytest.raises(ValueError, match=missing_message):
            handles_none._check_capabilities(missing, multivariate, unequal)
    if multivariate:
        with pytest.raises(ValueError, match=multivariate_message):
            handles_none._check_capabilities(missing, multivariate, unequal)
    if unequal:
        with pytest.raises(ValueError, match=unequal_message):
            handles_none._check_capabilities(missing, multivariate, unequal)
    if not missing and not multivariate and not unequal:
        handles_none._check_capabilities(missing, multivariate, unequal)

    handles_all = _DummyHandlesAllInput()
    handles_all._check_capabilities(missing, multivariate, unequal)


def test_score():
    dummy = DummyRegressor()
    x_train, y_train = load_unit_test(split="train")
    x_test, y_test = load_unit_test(split="test")
    dummy.fit(x_train, y_train)
    r = dummy.score(x_test, y_test)
    np.testing.assert_almost_equal(r, -0.008333, decimal=6)
