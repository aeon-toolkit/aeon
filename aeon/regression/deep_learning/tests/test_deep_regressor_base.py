"""Unit tests for regressors deep learning base class functionality."""
import gc
import os
import time

import pytest

from aeon.regression.deep_learning.base import BaseDeepRegressor
from aeon.utils._testing.collection import make_2d_test_data
from aeon.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["achieveordie", "hadifawaz1999"]


class _DummyDeepRegressor(BaseDeepRegressor):
    """Dummy Deep Regressor for testing empty base deep class save utilities."""

    def __init__(self, last_file_name):
        self.last_file_name = last_file_name
        super(_DummyDeepRegressor, self).__init__(last_file_name=last_file_name)

    def build_model(self, input_shape):
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        gap = tf.keras.layers.GlobalAveragePooling1D()(input_layer)
        output_layer = tf.keras.layers.Dense(units=1)(gap)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss="mse")

        return model

    def _fit(self, X, y):
        X = X.transpose(0, 2, 1)

        self.input_shape_ = X.shape[1:]
        self.model_ = self.build_model(self.input_shape_)

        self.history = self.model_.fit(
            X,
            y,
            batch_size=16,
            epochs=2,
        )

        gc.collect()
        return self


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dummy_deep_regressor():
    last_file_name = str(time.time_ns())

    # create a dummy regressor
    dummy_deep_rg = _DummyDeepRegressor(last_file_name=last_file_name)

    # generate random data

    X, y = make_2d_test_data()

    # test fit function on random data
    dummy_deep_rg.fit(X=X, y=y)

    # test save last model to file than delete it

    dummy_deep_rg.save_last_model_to_file()

    os.remove("./" + last_file_name + ".hdf5")

    # test summary of model

    assert dummy_deep_rg.summary() is not None
