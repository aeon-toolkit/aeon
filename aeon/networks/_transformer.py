"""Transformer Network (TransformerNetwork)."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class TransformerNetwork(BaseDeepLearningNetwork):
    """Transformer Network.

    Parameters
    ----------
    n_layers : int, default = 2
        The number of transformer blocks.
    n_heads : int, default = 4
        The number of heads in the multi-head attention layer.
    d_model : int, default = 64
        The dimension of the embedding vector.
    d_inner : int, default = 128
        The dimension of the feed-forward network in the transformer block.
    activation : str, default = "relu"
        The activation function used in the feed-forward network.
    dropout : float, default = 0.1
        The dropout rate.

    Structure
    ---------
    It uses the standard Transformer Encoder architecture:
    1. Input Embedding (Conv1D)
    2. Positional Encoding
    3. N x Transformer Encoder Blocks (MultiHeadAttention -> Add & Norm -> FeedForward
       -> Add & Norm)
    4. Global Average Pooling
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_inner=128,
        activation="relu",
        dropout=0.1,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_inner = d_inner
        self.activation = activation
        self.dropout = dropout

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
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        input_layer = keras.Input(shape=input_shape)

        # 1. Input Embedding / Projection
        # Project input channels to d_model dimension
        x = layers.Conv1D(filters=self.d_model, kernel_size=1, padding="same")(
            input_layer
        )
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # 2. Positional Encoding
        # We'll use a simple learnable position embedding here for simplicity
        # and compatibility
        # Sequence length is input_shape[0]
        seq_len = input_shape[0]

        # Create positions
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embedding = layers.Embedding(
            input_dim=seq_len, output_dim=self.d_model
        )(positions)
        x = x + position_embedding

        # 3. Transformer Encoder Blocks
        for _ in range(self.n_layers):
            # Attention
            # Pre-Norm architecture is often more stable
            x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
            attn_output = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=self.dropout,
            )(x_norm, x_norm)

            # Skip connection
            x = layers.Add()([x, attn_output])

            # Feed Forward
            x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
            ffn_output = layers.Dense(self.d_inner, activation=self.activation)(x_norm)
            ffn_output = layers.Dropout(self.dropout)(ffn_output)
            ffn_output = layers.Dense(self.d_model)(ffn_output)

            # Skip connection
            x = layers.Add()([x, ffn_output])

        # 4. Global Average Pooling
        output_layer = layers.GlobalAveragePooling1D()(x)

        return input_layer, output_layer
