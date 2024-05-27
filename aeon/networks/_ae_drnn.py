"""Auto-Encoder based Dilated Recurrent Neural Networks (DRNN)."""

__maintainer__ = []

from aeon.networks.base import BaseDeepNetwork


class AEDRNNNetwork(BaseDeepNetwork):
    """Auto-Encoder based Dilated Recurrent Neural Networks (DRNN).

    Parameters
    ----------
    latent_space_dim : int, default = 128
        Dimensionality of the latent space.
    n_layers : int, default = 3
        Number of GRU layers in the encoder.
    dilation_rate : List[int], default = None
        List of dilation rates for each layer, by default None.
        If None, default = powers of 2 up to `n_stacked`.
    activation : str, default="relu"
        Activation function to use in the GRU layers.
    decoder_activation : str, default=None
        Activation function of the single GRU layer in the decoder.
    n_units : List[int], default="None"
        Number of units in each GRU layer, by default None.
        If None, default to [100, 50, 50].
    """

    def __init__(
        self,
        latent_space_dim=128,
        n_layers=3,
        dilation_rate=None,
        activation="relu",
        decoder_activation=None,
        n_units=None,
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.n_units = n_units
        self.activation = activation
        self.decoder_activation = decoder_activation
        self.n_layers = n_layers
        self.dilation_rate = dilation_rate

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

        if self.decoder_activation is None:
            self.decoder_activation = self.activation
        
        if isinstance(self.decoder_activation, list):
            self.decoder_activation = self.activation[0]

        if self.dilation_rate is None:
            self.dilation_rate = [2**i for i in range(self.n_layers)]
        else:
            assert isinstance(self.dilation_rate, list)
            assert len(self.dilation_rate) == self.n_layers

        if self.n_units is None:
            assert self.n_layers == 3
            self.n_units = [100, 50, 50]
        else:
            assert isinstance(self.n_units, list)
            assert len(self.n_units) == self.n_layers

        if isinstance(self.activation, str):
            self.activation = [self.activation for _ in range(self.n_layers)]
        elif isinstance(self.activation, list):
            assert len(self.activation) == self.n_layers

        encoder_input_layer = tf.keras.layers.Input(input_shape)
        x = encoder_input_layer

        _finals = []

        for i in range(self.n_layers - 1):
            final, output = self._bidir_gru(
                x,
                self.n_units[i],
                activation=self.activation[i],
            )
            x = tf.keras.layers.Lambda(
                self._dilate_input, arguments={"dilation_rate": self.dilation_rate[i]}
            )(output)
            _finals.append(final)

        final, output = self._bidir_gru(x, self.n_units[-1], activation=self.activation[-1])
        _finals.append(final)
        _output = tf.keras.layers.Concatenate()(_finals)

        encoder_output_layer = tf.keras.layers.Dense(
            self.latent_space_dim, activation="linear"
        )(_output)

        encoder = tf.keras.Model(
            inputs=encoder_input_layer, outputs=encoder_output_layer
        )

        decoder_input_layer = tf.keras.layers.Input(shape=(self.latent_space_dim,))
        expanded_latent_space = tf.keras.layers.RepeatVector(input_shape[0])(
            decoder_input_layer
        )

        decoder_gru_units = sum(self.n_units) * 2
        decoder_gru = tf.keras.layers.GRU(
            decoder_gru_units, return_sequences=True, activation=self.decoder_activation,
        )(expanded_latent_space)

        decoder_output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(input_shape[1], activation="linear")
        )(decoder_gru)

        decoder = tf.keras.Model(
            inputs=decoder_input_layer, outputs=decoder_output_layer
        )

        return encoder, decoder

    def _dilate_input(self, tensor, dilation_rate):
        return tensor[:, ::dilation_rate, :]

    def _bidir_gru(self, input, nunits, activation):
        import tensorflow as tf

        output, forward, backward = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                nunits,
                activation=activation,
                return_sequences=True,
                return_state=True,
            )
        )(input)
        final = tf.keras.layers.Concatenate()([forward, backward])
        return final, output
