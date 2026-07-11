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


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_prepare_callbacks_variants():
    """User callbacks are merged with the checkpoint callback."""
    import tensorflow as tf

    cb = tf.keras.callbacks.History()

    single = DummyDeepForecaster(window=5)
    single.callbacks = cb
    cb_list = single._prepare_callbacks()
    assert cb in cb_list and len(cb_list) == 2

    listed = DummyDeepForecaster(window=5)
    listed.callbacks = [cb]
    cb_list = listed._prepare_callbacks()
    assert cb in cb_list and len(cb_list) == 2

    # non-list input to the checkpoint helper is wrapped
    wrapped = single._get_model_checkpoint_callback(cb, "./", "tmp_model")
    assert isinstance(wrapped, list) and len(wrapped) == 2


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_summary_save_load_and_test_params():
    """summary, save/load helpers and base test params work."""
    import os
    import tempfile

    import numpy as np
    import tensorflow as tf

    f = DummyDeepForecaster(window=5)
    assert f.summary() is None
    with pytest.raises(ValueError, match="No model to save"):
        f.save_last_model_to_file()

    params = DummyDeepForecaster._get_test_params()
    assert params[0]["window"] == 10

    # attach a minimal keras model to exercise save/load round trip
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(3,))])
    model.compile(loss="mse")
    f.model_ = model
    with tempfile.TemporaryDirectory() as tmp:
        f.save_last_model_to_file(file_path=tmp)
        path = os.path.join(tmp, f.last_file_name + ".keras")
        assert os.path.exists(path)
        f2 = DummyDeepForecaster(window=5)
        f2.load_model(path)
        assert f2.is_fitted and f2.model_ is not None
        pred = f2.model_.predict(np.zeros((1, 3)), verbose=0)
        assert pred.shape == (1, 1)
