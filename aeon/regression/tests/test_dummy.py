"""Test function of DummyRegressor."""

import numpy as np

from aeon.datasets import load_covid_3month
from aeon.regression import DummyRegressor


def test_dummy_regressor():
    """Test function for DummyRegressor."""
    X_train, y_train = load_covid_3month(split="train")
    X_test, _ = load_covid_3month(split="test")
    dummy = DummyRegressor()
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    assert np.all(np.isclose(pred, 0.03689763))
