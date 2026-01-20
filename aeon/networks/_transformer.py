"""Transformer Network (TransformerNetwork)."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


def _get_ape_class():
    """Get the APE (Absolute Positional Encoder) class.

    This is defined inside a function to avoid top-level TensorFlow usage,
    which breaks core dependency tests.
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class APE(layers.Layer):
        """Compute Absolute positional encoder."""

        def __init__(
            self,
            d_model: int,
            dropout_rate: float = 0.1,
            max_len: int = 1024,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.d_model = d_model
            self.dropout_rate = dropout_rate
            self.max_len = max_len
            self.dropout = layers.Dropout(dropout_rate)

            position = np.arange(max_len)[:, np.newaxis]
            div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

            pe = np.zeros((max_len, d_model))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            # Add newaxis to make it (1, max_len, d_model) for broadcasting
            self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

        def build(self, input_shape):
            super().build(input_shape)

        def call(self, x):
            x += self.pe[:, : tf.shape(x)[1], :]
            return self.dropout(x)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "d_model": self.d_model,
                    "dropout_rate": self.dropout_rate,
                    "max_len": self.max_len,
                }
            )
            return config

    return APE


class TransformerNetwork(BaseDeepLearningNetwork):
    """Transformer Network.

    Parameters
    ----------
    n_layers : int, default = 4
        The number of transformer encoder layers.
    n_heads : int, default = 4
        The number of heads in the multi-head attention layer.
    d_model : int, default = 256
        The dimension of the embedding vector.
    d_inner : int, default = 1024
        The dimension of the feed-forward network in the transformer block.
    activation : str, default = "gelu"
        The activation function used in the feed-forward network.
    dropout : float, default = 0.1
        The dropout rate for regularization.
    epsilon : float, default = 1e-6
        Small value to avoid division by zero in normalization layers.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_inner=1024,
        activation="gelu",
        dropout=0.1,
        epsilon=1e-6,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_inner = d_inner
        self.activation = activation
        self.dropout = dropout
        self.epsilon = epsilon

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras
        from tensorflow.keras import layers

        # Get the APE class lazily
        APE = _get_ape_class()

        input_layer = keras.Input(shape=input_shape)

        # 1. Input Embedding / Projection
        x = layers.Dense(units=self.d_model, activation="linear")(input_layer)

        # 2. Positional Encoding using APE
        x = APE(
            d_model=self.d_model,
            dropout_rate=self.dropout,
            max_len=input_shape[0],
        )(x)

        if isinstance(self.activation, list):
            if len(self.activation) != self.n_layers:
                raise ValueError(
                    f"Number of activations {len(self.activation)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_layers

        # 3. Transformer Encoder Blocks
        for i in range(self.n_layers):
            # Attention
            mha = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model,
            )(query=x, value=x, key=x)

            x_dropped = layers.Dropout(self.dropout)(mha)
            x_norm = layers.LayerNormalization(epsilon=self.epsilon)(x_dropped)
            res = layers.Add()([x, x_norm])

            # Feed Forward
            ffn = layers.Dense(units=self.d_inner, activation=self._activation[i])(res)
            ffn = layers.Dense(units=self.d_model, activation="linear")(ffn)

            x_dropped = layers.Dropout(self.dropout)(ffn)
            x_norm = layers.LayerNormalization(epsilon=self.epsilon)(x_dropped)
            x = layers.Add()([res, x_norm])

        # 4. Global Average Pooling
        output_layer = layers.GlobalAveragePooling1D()(x)

        return input_layer, output_layer
