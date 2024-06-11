"""Unit tests for classifiers deep learning base class functionality."""

import gc
import tempfile
import time

import pytest

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.testing.data_generation import make_example_2d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = []


class _DummyDeepClassifier(BaseDeepClassifier):
    """Dummy Deep Classifier for testing empty base deep class save utilities."""

    def __init__(self, last_file_name="last_model"):
        self.last_file_name = last_file_name
        super().__init__(last_file_name=last_file_name)

    def build_model(self, input_shape, n_classes):
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        gap = tf.keras.layers.GlobalAveragePooling1D()(input_layer)
        output_layer = tf.keras.layers.Dense(units=n_classes, activation="softmax")(gap)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=tf.keras.losses.categorical_crossentropy)

        return model

    def _fit(self, X, y):
        X = X.transpose(0, 2, 1)

        # test convert y to one hot vector
        y_onehot = self.convert_y_to_keras(y)

        self.input_shape_ = X.shape[1:]
        self.model_ = self.build_model(self.input_shape_, self.n_classes_)

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=16,
            epochs=2,
        )

        gc.collect()
        return self


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dummy_deep_classifier():
    """Test dummy deep classifier."""
    with tempfile.TemporaryDirectory() as tmp:
        import numpy as np

        last_file_name = str(time.time_ns())

        # create a dummy deep classifier
        dummy_deep_clf = _DummyDeepClassifier(last_file_name=last_file_name)

        # generate random data
        X, y = make_example_2d_numpy()

        # test fit function on random data
        dummy_deep_clf.fit(X=X, y=y)

        # test save last model to file than delete it
        dummy_deep_clf.save_last_model_to_file(file_path=tmp)

        # create a new dummy deep classifier
        dummy_deep_clf2 = _DummyDeepClassifier()

        # load without fitting
        dummy_deep_clf2.load_model(
            model_path=tmp + last_file_name + ".keras", classes=np.unique(y)
        )

        # predict
        ypred = dummy_deep_clf2.predict(X=X)

        assert len(ypred) == len(y)

        # test summary of model
        assert dummy_deep_clf.summary() is not None
