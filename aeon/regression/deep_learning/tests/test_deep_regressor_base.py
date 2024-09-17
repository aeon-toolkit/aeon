"""Unit tests for regressors deep learning base class functionality."""

import gc
import tempfile
import time

import pytest

from aeon.regression.deep_learning.base import BaseDeepRegressor
from aeon.testing.data_generation import make_example_2d_numpy_collection
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = ["hadifawaz1999"]


class _DummyDeepRegressor(BaseDeepRegressor):
    """Dummy Deep Regressor for testing empty base deep class save utilities."""

    def __init__(self, last_file_name="last_model"):
        self.last_file_name = last_file_name
        super().__init__(last_file_name=last_file_name)

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
    """Test for DummyRegressor."""
    with tempfile.TemporaryDirectory() as tmp:
        last_file_name = str(time.time_ns())

        # create a dummy regressor
        dummy_deep_rg = _DummyDeepRegressor(last_file_name=last_file_name)

        # generate random data

        X, y = make_example_2d_numpy_collection()

        # test fit function on random data
        dummy_deep_rg.fit(X=X, y=y)

        # test save last model to file than delete it

        dummy_deep_rg.save_last_model_to_file(file_path=tmp)

        # create a new dummy deep classifier
        dummy_deep_clf2 = _DummyDeepRegressor()

        # load without fitting
        dummy_deep_clf2.load_model(model_path=tmp + last_file_name + ".keras")

        # predict
        ypred = dummy_deep_clf2.predict(X=X)

        assert len(ypred) == len(y)

        # test summary of model

        assert dummy_deep_rg.summary() is not None
