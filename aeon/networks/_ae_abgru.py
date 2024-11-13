"""Implements an Auto-Encoder based on Attention Bidirectional GRUs."""

__maintainer__ = ["aadya940", "hadifawaz1999"]

from aeon.networks.base import BaseDeepLearningNetwork


class AEAttentionBiGRUNetwork(BaseDeepLearningNetwork):
    """
    A class to implement an Auto-Encoder based on Attention Bidirectional GRUs.

    Parameters
    ----------
        latent_space_dim : int, default=128
            Dimension of the latent space.
        temporal_latent_space : bool, default=False
            Flag to choose whether the latent space is an MTS or Euclidean space.
        n_layers_encoder : int, default=None
            Number of Attention BiGRU layers in the encoder.
            If None, one layer will be used.
        n_layers_decoder : int, default=None
            Number of Attention BiGRU layers in the decoder.
            If None, one layer will be used.
        activation_encoder : Union[list, str], default="relu"
            Activation function(s) to use in each layer of the encoder.
            Can be a single string or a list.
        activation_decoder : Union[list, str], default="relu"
            Activation function(s) to use in each layer of the decoder.
            Can be a single string or a list.

    References
    ----------
    .. [1] Ienco, D., & Interdonato, R. (2020). Deep multivariate time series
    embedding clustering via attentive-gated autoencoder. In Advances in Knowledge
    Discovery and Data Mining: 24th Pacific-Asia Conference, PAKDD 2020, Singapore,
    May 11-14, 2020, Proceedings, Part I 24 (pp. 318-329). Springer International
    Publishing.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        latent_space_dim=None,
        temporal_latent_space=False,
        n_layers_encoder=1,
        n_layers_decoder=1,
        activation_encoder="relu",
        activation_decoder="relu",
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.temporal_latent_space = temporal_latent_space
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.
        kwargs : dict
            Contains additional keyword arguments to be passed to the
            `build_network` method like `num_input_samples`.

        Returns
        -------
        encoder : a keras Model.
        decoder : a keras Model.
        """
        import tensorflow as tf

        if isinstance(self.activation_encoder, str):
            self._activation_encoder = [
                self.activation_encoder for _ in range(self.n_layers_encoder)
            ]
        else:
            self._activation_encoder = self.activation_encoder
            if not isinstance(self.activation_encoder, list):
                raise ValueError(
                    "Encoder activations should be a list or a single string."
                )
            if len(self.activation_encoder) != self.n_layers_encoder:
                raise ValueError(
                    f"Number of encoder activations {len(self.activation_encoder)}"
                    f" should be same as number of encoder layers but is"
                    f" not: {self.n_layers_encoder}"
                )

        if isinstance(self.activation_decoder, str):
            self._activation_decoder = [
                self.activation_decoder for _ in range(self.n_layers_decoder)
            ]
        else:
            self._activation_decoder = self.activation_decoder
            if not isinstance(self.activation_decoder, list):
                raise ValueError(
                    "Decoder activations should be a list or a single string."
                )
            if len(self.activation_decoder) != self.n_layers_decoder:
                raise ValueError(
                    f"Number of decoder activations {len(self.activation_decoder)}"
                    f" should be same as number of decoder layers but is"
                    f" not: {self.n_layers_decoder}"
                )

        if not isinstance(self.n_layers_encoder, int):
            raise ValueError("Number of layers of the encoder must be an integer.")

        if not isinstance(self.n_layers_decoder, int):
            raise ValueError("Number of layers of the decoder must be an integer.")

        input_layer = tf.keras.layers.Input(input_shape)
        x = input_layer

        if self.latent_space_dim is None:
            if "num_input_samples" in kwargs.keys():
                self.n_filters_RNN = 64
                if kwargs["input_samples"] > 250:
                    self.n_filters_RNN = 512
            else:
                self.latent_space_dim = 64
                self.n_filters_RNN = self.latent_space_dim
        elif self.latent_space_dim is not None:
            self.n_filters_RNN = self.latent_space_dim

        for i in range(self.n_layers_encoder):
            forward_layer = tf.keras.layers.GRU(
                self.n_filters_RNN,
                activation=self._activation_encoder[i],
                return_sequences=True,
            )(x)
            backward_layer = tf.keras.layers.GRU(
                self.n_filters_RNN,
                activation=self._activation_encoder[i],
                return_sequences=True,
                go_backwards=True,
            )(x)

            query = tf.keras.layers.Dense(self.n_filters_RNN)(forward_layer)
            key = tf.keras.layers.Dense(self.n_filters_RNN)(backward_layer)
            value = tf.keras.layers.Dense(self.n_filters_RNN)(backward_layer)

            attention_layer = tf.keras.layers.Attention()([query, key, value])
            x = tf.keras.layers.Dense(self.n_filters_RNN, activation="sigmoid")(
                attention_layer
            )
            x = x * attention_layer

        if not self.temporal_latent_space:
            shape_before_flatten = x.shape[1:]
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(self.latent_space_dim)(x)
        elif self.temporal_latent_space:
            x = tf.keras.layers.Conv1D(filters=self.latent_space_dim, kernel_size=1)(x)

        encoder = tf.keras.models.Model(inputs=input_layer, outputs=x, name="encoder")

        if not self.temporal_latent_space:
            decoder_inputs = tf.keras.layers.Input(
                shape=(self.latent_space_dim,), name="decoder_input"
            )
            x = tf.keras.layers.RepeatVector(input_shape[0], name="repeat_vector")(
                decoder_inputs
            )
        else:
            decoder_inputs = tf.keras.layers.Input(
                shape=shape_before_flatten, name="decoder_input"
            )
            x = decoder_inputs

        for i in range(self.n_layers_decoder - 1, -1, -1):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=self.n_filters_RNN // 2,
                    activation=self._activation_decoder[i],
                    return_sequences=True,
                ),
                name=f"decoder_bgru_{i+1}",
            )(x)

        decoder_outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(input_shape[1]), name="decoder_output"
        )(x)
        decoder = tf.keras.models.Model(
            inputs=decoder_inputs, outputs=decoder_outputs, name="decoder"
        )

        return encoder, decoder
