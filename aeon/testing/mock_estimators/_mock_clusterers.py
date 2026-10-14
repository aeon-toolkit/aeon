"""Mock clusterers useful for testing and debugging."""

__maintainer__ = []
__all__ = [
    "MockCluster",
    "MockTransductiveCluster",
    "MockDeepClusterer",
]

import numpy as np

from aeon.clustering.base import BaseClusterer
from aeon.clustering.deep_learning.base import BaseDeepClusterer


class MockCluster(BaseClusterer):
    """Mock Cluster for testing base class fit/predict."""

    def __init__(self):
        super().__init__()

    def _fit(self, X):
        """Mock fit."""
        self.labels_ = np.zeros(len(X), dtype=int)
        return self

    def _predict(self, X):
        """Mock predict."""
        return np.zeros(len(X))

    def _predict_proba(self, X):
        """Mock predict proba."""
        y = np.random.rand(len(X))
        return y


class MockTransductiveCluster(BaseClusterer):
    """Mock transductive clusterer without out-of-sample prediction."""

    _tags = {
        "capability:predict": False,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X):
        """Mock fit."""
        self.labels_ = np.arange(len(X)) % 2
        return self

    def _predict(self, X):
        """Mock predict, unreachable through the public API."""
        raise RuntimeError(
            "predict should not be reachable for transductive clusterers."
        )


class MockDeepClusterer(BaseDeepClusterer):
    """Mock Deep Clusterer for testing empty base deep class save utilities."""

    def __init__(self, estimator=None, last_file_name="last_file"):
        self.last_file_name = last_file_name
        super().__init__(
            estimator=estimator,
            last_file_name=last_file_name,
        )

    def build_model(self, input_shape):
        """Build a Mock model."""
        import tensorflow as tf

        self._check_params()

        input_layer_encoder = tf.keras.layers.Input(input_shape)
        gap = tf.keras.layers.GlobalAveragePooling1D()(input_layer_encoder)
        output_layer_encoder = tf.keras.layers.Dense(units=10)(gap)
        encoder = tf.keras.models.Model(
            inputs=input_layer_encoder, outputs=output_layer_encoder, name="encoder"
        )

        input_layer_decoder = tf.keras.layers.Input((10,))
        dense = tf.keras.layers.Dense(10)(input_layer_decoder)
        # Cast to int to avoid Keras rejecting numpy scalar types
        decoder_units = int(np.prod(input_shape))
        _output_layer_decoder = tf.keras.layers.Dense(decoder_units)(dense)
        output_layer_decoder = tf.keras.layers.Reshape(target_shape=input_shape)(
            _output_layer_decoder
        )

        decoder = tf.keras.models.Model(
            inputs=input_layer_decoder, outputs=output_layer_decoder, name="decoder"
        )

        input_layer = tf.keras.layers.Input(input_shape)
        encoder_ouptut = encoder(input_layer)
        decoder_output = decoder(encoder_ouptut)

        model = tf.keras.models.Model(inputs=input_layer, outputs=decoder_output)

        model.compile(loss="mse")

        return model

    def _score(self, X, y=None):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        return self.estimator_.score(latent_space)
