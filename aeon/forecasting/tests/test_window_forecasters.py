"""Tests for window forecasters."""

__maintainer__ = []

import numpy as np
import pytest

from aeon.forecasting.base._aeon import _BaseWindowForecaster
from aeon.forecasting.model_selection import temporal_train_test_split
from aeon.registry import all_estimators
from aeon.testing.utils.data_gen import make_forecasting_problem, make_series

FH0 = 1

WINDOW_FORECASTERS = [
    forecaster
    for (name, forecaster) in all_estimators(estimator_types="forecaster")
    if issubclass(forecaster, _BaseWindowForecaster)
]

# testing data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.parametrize("Forecaster", WINDOW_FORECASTERS)
def test_last_window(Forecaster):
    """Test window forecaster common API points."""
    f = Forecaster.create_test_instance()
    n_columns = 1

    f = Forecaster.create_test_instance()
    y_train = make_series(n_columns=n_columns)
    # passing the same fh to both fit and predict works
    f.fit(y_train, fh=FH0)

    actual, _ = f._get_last_window()
    expected = y_train.iloc[-f.window_length_ :]

    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == f.window_length_
