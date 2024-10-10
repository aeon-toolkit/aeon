"""Test for TimeCNN regressor."""

import pytest

from aeon.regression.deep_learning import TimeCNNRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_timecnn_inputs():
    """Test parameter inputs correctly handled."""
    X, y = make_example_3d_numpy(n_cases=3, n_channels=1, n_timepoints=10)
    cnn = TimeCNNRegressor(metrics=["mean_squared_error"], callbacks=None)
    cnn.fit(X, y)
