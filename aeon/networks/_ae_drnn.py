"""Auto-Encoder based Dilated Recurrent Neural Networks (DRNN)."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class AEDRNNNetwork(BaseDeepLearningNetwork):
    """Auto-Encoder based Dilated Recurrent Neural Networks (DRNN).

    Parameters
    ----------
    latent_space_dim : int, default = 128
        Dimensionality of the latent space.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    n_layers_encoder : int, default = 3
        Number of GRU layers in the encoder.
    n_layers_decoder : int, default = 1
        Number of GRU layers in the decoder.
    dilation_rate_encoder : Union[int, List[int]], default = None
        List of dilation rates for each layer of the encoder.
        If None, default = powers of 2 up to `n_stacked`.
    dilation_rate_decoder : Union[int, List[int]], default = None
        List of dilation rates for each layer of the decoder.
        If None, default to a list of ones.
    activation_encoder : Union[str, List[str]], default="relu"
        Activation function to use in the GRU layers.
    activation_decoder : Union[str, List[str]], default=None
        Activation function of the single GRU layer in the decoder.
        If None, defaults to relu.
    n_units_encoder : List[int], default="None"
        Number of units in each GRU layer of the encoder, by default None.
        If None, default to [100, 50, 50].
    n_units_decoder : List[int], default="None"
        Number of units in each GRU layer of the decoder, by default None.
        If None, default to two times sum of units of the encoder.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.12",
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        latent_space_dim=128,
        temporal_latent_space=False,
        n_layers_encoder=3,
        n_layers_decoder=1,
        dilation_rate_encoder=None,
        dilation_rate_decoder=None,
        activation_encoder="relu",
        activation_decoder=None,
        n_units_encoder=None,
        n_units_decoder=None,
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.n_units_encoder = n_units_encoder
        self.n_units_decoder = n_units_decoder
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.dilation_rate_encoder = dilation_rate_encoder
        self.dilation_rate_decoder = dilation_rate_decoder
        self.temporal_latent_space = temporal_latent_space

    def build_network(self, input_shape, **kwargs):
        """Build the encoder and decoder networks.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.
        **kwargs : dict
            Additional keyword arguments for building the network.

        Returns
        -------
        encoder : tf.keras.Model
            The encoder model.
        decoder : tf.keras.Model
            The decoder model.
        """
        import tensorflow as tf

        if self.activation_decoder is None:
            self._decoder_activation = ["relu" for _ in range(self.n_layers_decoder)]
        elif isinstance(self.activation_decoder, str):
            self._decoder_activation = [
                self.activation_decoder for _ in range(self.n_layers_decoder)
            ]
        elif isinstance(self.activation_decoder, list):
            self._decoder_activation = self.activation_decoder
            assert len(self._decoder_activation) == self.n_layers_decoder

        if self.dilation_rate_encoder is None:
            self._dilation_rate_encoder = [2**i for i in range(self.n_layers_encoder)]
        elif isinstance(self.dilation_rate_encoder, int):
            self._dilation_rate_encoder = [
                self.dilation_rate_encoder for _ in range(self.n_layers_encoder)
            ]
        else:
            self._dilation_rate_encoder = self.dilation_rate_encoder
            assert isinstance(self.dilation_rate_encoder, list)
            assert len(self.dilation_rate_encoder) == self.n_layers_encoder

        if self.n_units_encoder is None:
            if self.n_layers_encoder == 3:
                self._n_units_encoder = [100, 50, 50]
            else:
                self._n_units_encoder = [100] + [
                    50 for _ in range(self.n_layers_encoder - 1)
                ]
        else:
            self._n_units_encoder = self.n_units_encoder
            assert isinstance(self.n_units_encoder, list)
            assert len(self.n_units_encoder) == self.n_layers_encoder

        if self.n_units_decoder is None:
            self._n_units_decoder_ = sum(self._n_units_encoder) * 2
            self._n_units_decoder = [
                self._n_units_decoder_ for _ in range(self.n_layers_decoder)
            ]
        else:
            self._n_units_decoder = self.n_units_decoder
            assert isinstance(self.n_units_decoder, list)
            assert len(self.n_units_decoder) == self.n_layers_decoder

        if isinstance(self.activation_encoder, str):
            self._activation_encoder = [
                self.activation_encoder for _ in range(self.n_layers_encoder)
            ]
        elif isinstance(self.activation_encoder, list):
            self._activation_encoder = self.activation_encoder
            assert len(self.activation_encoder) == self.n_layers_encoder

        if self.dilation_rate_decoder is None:
            self._dilation_rate_decoder = [1 for _ in range(self.n_layers_decoder)]
        elif isinstance(self.dilation_rate_decoder, int):
            self._dilation_rate_decoder = [
                self.dilation_rate_decoder for _ in range(self.n_layers_decoder)
            ]
        elif isinstance(self.dilation_rate_decoder, list):
            self._dilation_rate_decoder = self.dilation_rate_decoder
            assert len(self._dilation_rate_decoder) == self.n_layers_decoder

        encoder_input_layer = tf.keras.layers.Input(input_shape)
        x = encoder_input_layer

        _finals = []

        for i in range(self.n_layers_encoder - 1):
            final, output = self._bidir_gru(
                x,
                self._n_units_encoder[i],
                activation=self._activation_encoder[i],
            )
            x = tf.keras.layers.Lambda(
                self._dilate_input,
                arguments={"dilation_rate": self._dilation_rate_encoder[i]},
            )(output)
            _finals.append(final)

        if not self.temporal_latent_space:
            final, output = self._bidir_gru(
                x,
                self._n_units_encoder[-1],
                activation=self._activation_encoder[-1],
                return_sequences=False,
            )
            _finals.append(final)
            _output = tf.keras.layers.Concatenate()(_finals)
            encoder_output_layer = tf.keras.layers.Dense(
                self.latent_space_dim, activation="linear"
            )(_output)

        elif self.temporal_latent_space:
            final, output = self._bidir_gru(
                x,
                self._n_units_encoder[-1],
                activation=self._activation_encoder[-1],
                return_sequences=True,
            )

            encoder_output_layer = tf.keras.layers.Conv1D(
                self.latent_space_dim,
                activation="linear",
                kernel_size=1,
            )(output)

        encoder = tf.keras.Model(
            inputs=encoder_input_layer, outputs=encoder_output_layer
        )

        if not self.temporal_latent_space:
            decoder_input_layer = tf.keras.layers.Input(shape=(self.latent_space_dim,))
            expanded_latent_space = tf.keras.layers.RepeatVector(input_shape[0])(
                decoder_input_layer
            )
        elif self.temporal_latent_space:
            decoder_input_layer = tf.keras.layers.Input(
                shape=encoder_output_layer.shape[1:]
            )
            expanded_latent_space = decoder_input_layer

        for i in range(self.n_layers_decoder):
            decoder_gru = tf.keras.layers.GRU(
                self._n_units_decoder[i],
                return_sequences=True,
                activation=self._decoder_activation[i],
            )(expanded_latent_space)
            if i < self.n_layers_decoder - 1:
                decoder_gru = tf.keras.layers.Lambda(
                    self._dilate_input,
                    arguments={"dilation_rate": self._dilation_rate_decoder[i]},
                )

        decoder_output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(input_shape[1], activation="linear")
        )(decoder_gru)

        decoder = tf.keras.Model(
            inputs=decoder_input_layer, outputs=decoder_output_layer
        )

        return encoder, decoder

    def _dilate_input(self, tensor, dilation_rate):
        return tensor[:, ::dilation_rate, :]

    def _bidir_gru(self, input, nunits, activation, return_sequences=True):
        import tensorflow as tf

        bidir_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                nunits,
                activation=activation,
                return_sequences=True,
                return_state=True,
            )
        )

        if return_sequences:
            output, forward_h, backward_h = bidir_gru(input)
        else:
            output, forward_h, backward_h = bidir_gru(input)
            output = output[
                :, -1, :
            ]  # Select the last output if not returning sequences

        final_state = tf.keras.layers.Concatenate()([forward_h, backward_h])
        return final_state, output