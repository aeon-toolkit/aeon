"""Test file for BaseDeepForecaster."""

import pytest

from aeon.forecasting.deep_learning.base import BaseDeepForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


class DummyDeepForecaster(BaseDeepForecaster):
    """Minimal concrete subclass to allow instantiation."""

    def __init__(self, window):
        super().__init__(window=window)

    def _predict(self, y, exog=None):
        return None

    def build_model(self, input_shape):
        """Construct and return a model based on the provided input shape."""
        return None  # Not needed for this test


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_default_init_attributes():
    """Test that BaseDeepForecaster sets default params and attributes correctly."""
    forecaster = DummyDeepForecaster(window=10)

    # check default parameters
    assert forecaster.horizon == 1
    assert forecaster.window == 10
    assert forecaster.verbose == 0
    assert forecaster.callbacks is None
    assert forecaster.axis == 0
    assert forecaster.last_file_name == "last_model"
    assert forecaster.file_path == "./"

    # check default attributes after init
    assert forecaster.model_ is None
    assert forecaster.history_ is None
    assert forecaster.last_window_ is None

    # check tags
    tags = forecaster.get_tags()
    assert tags["algorithm_type"] == "deeplearning"
    assert tags["capability:horizon"]
    assert tags["capability:univariate"]
