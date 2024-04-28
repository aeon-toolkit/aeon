import numpy as np

from aeon.clustering.deep_learning.base import BaseDeepClusterer


class MockDeepClusterer(BaseDeepClusterer):
    """Mock Deep Clusterer for testing empty base deep class save utilities."""

    def __init__(self, last_file_name="last_file"):
        self.last_file_name = last_file_name
        super().__init__(
            n_clusters=2,
            last_file_name=last_file_name,
            clustering_params={"n_init": 1, "averaging_method": "mean"},
        )

    def build_model(self, input_shape):
        """Build a Mock model."""
        import tensorflow as tf

        input_layer_encoder = tf.keras.layers.Input(input_shape)
        gap = tf.keras.layers.GlobalAveragePooling1D()(input_layer_encoder)
        output_layer_encoder = tf.keras.layers.Dense(units=10)(gap)
        encoder = tf.keras.models.Model(
            inputs=input_layer_encoder, outputs=output_layer_encoder
        )

        input_layer_decoder = tf.keras.layers.Input((10,))
        dense = tf.keras.layers.Dense(10)(input_layer_decoder)
        _output_layer_decoder = tf.keras.layers.Dense(np.prod(input_shape))(dense)
        output_layer_decoder = tf.keras.layers.Reshape(target_shape=input_shape)(
            _output_layer_decoder
        )

        decoder = tf.keras.models.Model(
            inputs=input_layer_decoder, outputs=output_layer_decoder
        )

        input_layer = tf.keras.layers.Input(input_shape)
        encoder_ouptut = encoder(input_layer)
        decoder_output = decoder(encoder_ouptut)

        model = tf.keras.models.Model(inputs=input_layer, outputs=decoder_output)

        model.compile(loss="mse")

        return model

    def _fit(self, X, y=None):
        X = X.transpose(0, 2, 1)

        self.input_shape_ = X.shape[1:]
        self.model_ = self.build_model(self.input_shape_)

        self.history = self.model_.fit(
            X,
            X,
            batch_size=16,
            epochs=1,
        )
        self._fit_clustering(X=X)

        #        gc.collect()
        return self

    def _score(self, X, y=None):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        return self.clusterer.score(latent_space)
