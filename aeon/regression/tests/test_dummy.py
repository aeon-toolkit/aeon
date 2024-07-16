"""Test function of DummyRegressor."""

import numpy as np
import pytest

from aeon.regression import DummyRegressor


@pytest.mark.parametrize("strategy", ["mean", "median", "quantile", "constant"])
def test_dummy_regressor_strategies(strategy):
    """Test DummyRegressor strategies."""
    X = np.ones(shape=(10, 10))
    y_train = np.random.rand(10)

    dummy = DummyRegressor(strategy=strategy, constant=0.5, quantile=0.5)
    dummy.fit(X, y_train)

    pred = dummy.predict(X)
    assert isinstance(pred, np.ndarray)
    assert all(0 <= i <= 1 for i in pred)


def test_dummy_regressor_default():
    """Test function for DummyRegressor."""
    X = np.ones(shape=(10, 10))
    y_train = np.array([1.5, 2, 1, 4, 5, 1, 1, 1.5, 0, 0.5])

    dummy = DummyRegressor()
    dummy.fit(X, y_train)

    pred = dummy.predict(X)
    assert np.all(np.isclose(pred, 1.75))
