"""Implement Auto-Encoder based on Bidirectional GRUs."""

from aeon.networks.base import BaseDeepNetwork


class AEBiGRUNetwork(BaseDeepNetwork):
    """
    A class to implement an Auto-Encoder based on Bidirectional GRUs.

    Parameters
    ----------
        latent_space_dim : int, default=128
            Dimension of the latent space.
        n_layers : int, default=None
            Number of BiGRU layers. If None, defaults will be used.
        n_units : list
            Number of units in each BiGRU layer. If None, defaults will be used.
        activation : Union[list, str]
            Activation function(s) to use in each layer.
            Can be a single string or a list.
        temporal_latent_space : bool, default = False
            Flag to choose whether the latent space is an MTS or Euclidean space.
    """

    def __init__(
        self,
        latent_space_dim=128,
        n_layers=None,
        n_units=None,
        activation="relu",
        temporal_latent_space=False,
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.activation = activation
        self.n_layers = n_layers
        self.n_units = n_units
        self.temporal_latent_space = temporal_latent_space

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        encoder : a keras Model.
        decoder : a keras Model.
        """
        from tensorflow.keras.layers import (
            GRU,
            Bidirectional,
            Dense,
            Input,
            RepeatVector,
            TimeDistributed,
        )
        from tensorflow.keras.models import Model

        if (self.n_layers is None) and (self.n_units is None):
            self._n_layers = 2
            self._n_units = [50, self.latent_space_dim // 2]
        elif (self.n_layers is not None) and (self.n_units is None):
            raise ValueError("Please specify the number of units for each GRU Layer.")
        else:
            self._n_layers = self.n_layers
            self._n_units = self.n_units

        if isinstance(self.activation, str):
            self._activation = [self.activation for _ in range(self.n_layers)]
        else:
            self._activation = self.activation
            assert isinstance(self.activation, list)
            assert len(self.activation) == self.n_layers

        encoder_inputs = Input(shape=input_shape, name="encoder_input")
        x = encoder_inputs
        for i in range(self._n_layers):
            return_sequences = i < self._n_layers - 1
            if self.temporal_latent_space:
                return_sequences = i < self._n_layers
            x = Bidirectional(
                GRU(
                    units=self._n_units[i],
                    activation=self._activation[i],
                    return_sequences=return_sequences,
                ),
                name=f"encoder_bgru_{i+1}",
            )(x)

        latent_space = Dense(
            self.latent_space_dim, activation="linear", name="latent_space"
        )(x)
        encoder_model = Model(
            inputs=encoder_inputs, outputs=latent_space, name="encoder"
        )

        if not self.temporal_latent_space:
            decoder_inputs = Input(shape=(self.latent_space_dim,), name="decoder_input")
            x = RepeatVector(input_shape[0], name="repeat_vector")(decoder_inputs)
        elif self.temporal_latent_space:
            decoder_inputs = Input(shape=latent_space.shape[1:], name="decoder_input")
            x = decoder_inputs

        for i in range(self._n_layers - 1, -1, -1):
            x = Bidirectional(
                GRU(
                    units=self._n_units[i],
                    activation=self._activation[i],
                    return_sequences=True,
                ),
                name=f"decoder_bgru_{i+1}",
            )(x)
        decoder_outputs = TimeDistributed(Dense(input_shape[1]), name="decoder_output")(
            x
        )
        decoder_model = Model(
            inputs=decoder_inputs, outputs=decoder_outputs, name="decoder"
        )

        return encoder_model, decoder_model
