"""Implements an Auto-Encoder based on Attention Bidirectional GRUs."""

__maintainer__ = []

from aeon.networks.base import BaseDeepAENetwork
from aeon.typing import LATENT_SPACE
from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class AdditiveAttentionLayer(tf.keras.layers.Layer):
        """Soft attention layer for BiGRU autoencoder."""

        def __init__(self, hidden_size, **kwargs):
            super().__init__(**kwargs)

            self.hidden_size = hidden_size

            self.projection = tf.keras.layers.Dense(
                hidden_size,
                activation="tanh",
            )

            self.context = self.add_weight(
                shape=(hidden_size,),
                initializer="glorot_uniform",
                trainable=True,
                name="context",
            )

        def call(self, inputs):
            # inputs: (batch, time, hidden_size)
            v = self.projection(inputs)
            scores = tf.reduce_sum(v * self.context, axis=-1)
            weights = tf.nn.softmax(scores, axis=1)
            return tf.reduce_sum(
                inputs * weights[..., None],
                axis=1,
            )

        def get_config(self):
            config = super().get_config()
            config.update({"hidden_size": self.hidden_size})
            return config


class AEDeTSECNetwork(BaseDeepAENetwork):
    """
    A class to implement an Auto-Encoder based on Attention Bidirectional GRUs.

    Parameters
    ----------
    n_filters_encoder : int, default=64
        Number of filters in the encoder.
    n_filters_decoder : int or list of int, default=64
        Number of filters in the decoder.
    activation_encoder : str, default='tanh'
        Activation function of the encoder.
    activation_decoder : str, default='tanh'
        Activation function of the decoder.

    References
    ----------
    .. [1] Ienco, D., & Interdonato, R. (2020). Deep multivariate time series
    embedding clustering via attentive-gated autoencoder. In Advances in Knowledge
    Discovery and Data Mining: 24th Pacific-Asia Conference, PAKDD 2020, Singapore,
    May 11-14, 2020, Proceedings, Part I 24 (pp. 318-329). Springer International
    Publishing.
    """

    _config = {
        **BaseDeepAENetwork._config,
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        n_filters_encoder=64,
        n_filters_decoder=64,
        activation_encoder="tanh",
        activation_decoder="tanh",
    ):
        super().__init__(0, LATENT_SPACE.REPEATED)
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.n_filters_encoder = n_filters_encoder
        self.n_filters_decoder = n_filters_decoder

    def _check_params(self):
        pass  # no parameters to check

    def _build_encoder_graph(self, x):
        import tensorflow as tf

        self._input_shape = x.shape[1:]  # save for reconstruction

        forward_h = tf.keras.layers.GRU(
            self.n_filters_encoder,
            activation=self.activation_encoder,
            return_sequences=True,
        )(x)
        backward_h = tf.keras.layers.GRU(
            self.n_filters_encoder,
            activation=self.activation_encoder,
            return_sequences=True,
            go_backwards=True,
        )(x)

        forward_embedding = AdditiveAttentionLayer(self.n_filters_encoder)(forward_h)
        backward_embedding = AdditiveAttentionLayer(self.n_filters_encoder)(backward_h)

        forward_gate = tf.keras.layers.Dense(
            self.n_filters_encoder, activation="sigmoid"
        )(forward_embedding)
        backward_gate = tf.keras.layers.Dense(
            self.n_filters_encoder, activation="sigmoid"
        )(backward_embedding)

        forward_embedding = forward_embedding * forward_gate
        backward_embedding = backward_embedding * backward_gate
        x = forward_embedding + backward_embedding
        return x

    def _build_latent_graph(self, x):
        self._enc_out = x
        self._dec_in = x
        x = tf.keras.layers.RepeatVector(self._input_shape[0])(x)
        return x

    def _build_decoder_graph(self, x):
        import tensorflow as tf

        forward_h = tf.keras.layers.GRU(
            self.n_filters_decoder,
            activation=self.activation_decoder,
            return_sequences=True,
        )(x)
        backward_h = tf.keras.layers.GRU(
            self.n_filters_decoder,
            activation=self.activation_decoder,
            return_sequences=True,
            go_backwards=True,
        )(x)

        forward_out = tf.keras.layers.Dense(self._input_shape[1])(forward_h)
        backward_out = tf.keras.layers.Dense(self._input_shape[1])(backward_h)
        backward_out = tf.keras.ops.flip(backward_out, axis=1)

        return forward_out, backward_out

    def build_base_graph(self, x):
        self._check_params()
        self._input_shape = x.shape[1:]  # save for reconstruction
        x = self._build_encoder_graph(x)
        x = self._build_latent_graph(x)
        x = self._build_decoder_graph(x)
        return x
