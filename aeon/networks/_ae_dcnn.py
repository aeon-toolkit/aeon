"""Auto-Encoder based on Dilated Convolutional Nerual Networks (DCNN) Model."""

__maintainer__ = []

import numpy as np

from aeon.networks.base import BaseDeepLearningNetwork


class AEDCNNNetwork(BaseDeepLearningNetwork):
    """Establish the Auto-Encoder based structure for a DCN Network.

    Dilated Convolutional Neural (DCN) Network based Model
    for low-rank embeddings.

    Parameters
    ----------
    latent_space_dim: int, default=128
        Dimension of the models's latent space.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    n_layers: int, default=4
        Number of convolution layers in the autoencoder.
    kernel_size: Union[int, List[int]], default=3
        Size of the 1D Convolutional Kernel of the encoder.
    activation: Union[str, List[str]], default="relu"
        The activation function used by convolution layers of the encoder.
    n_filters: Union[int, List[int]], default=None
        Number of filters used in convolution layers of the encoder.
    dilation_rate: Union[int, List[int]], default=None
        The dilation rate for convolution of the encoder.

    References
    ----------
    .. [1] Franceschi, J. Y., Dieuleveut, A., & Jaggi, M. (2019). Unsupervised
    scalable representation learning for multivariate time series. Advances in
    neural information processing systems, 32.

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
        n_layers=4,
        kernel_size=3,
        activation="relu",
        n_filters=None,
        dilation_rate=None,
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.temporal_latent_space = temporal_latent_space

    def build_network(self, input_shape):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        model : a keras Model.
        """
        import tensorflow as tf

        if self.n_filters is None:
            self._n_filters_encoder = [32 * i for i in range(1, self.n_layers + 1)]
        elif isinstance(self.n_filters, list):
            self._n_filters_encoder = self.n_filters
            assert len(self.n_filters) == self.n_layers

        if self.dilation_rate is None:
            self._dilation_rate_encoder = [
                2**layer_num for layer_num in range(1, self.n_layers + 1)
            ]
        elif isinstance(self.dilation_rate, int):
            self._dilation_rate_encoder = [
                self.dilation_rate for _ in range(self.n_layers)
            ]
        else:
            self._dilation_rate_encoder = self.dilation_rate
            assert isinstance(self.dilation_rate, list)
            assert len(self.dilation_rate) == self.n_layers

        if self.kernel_size is None:
            self._kernel_size_encoder = [3 for _ in range(self.n_layers)]
        elif isinstance(self.kernel_size, int):
            self._kernel_size_encoder = [self.kernel_size for _ in range(self.n_layers)]
        elif isinstance(self.kernel_size, list):
            self._kernel_size_encoder = self.kernel_size
            assert len(self.kernel_size) == self.n_layers

        if isinstance(self.activation, str):
            self._activation_encoder = [self.activation for _ in range(self.n_layers)]
        elif isinstance(self.activation, list):
            self._activation_encoder = self.activation
            assert len(self._activation_encoder) == self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(0, self.n_layers):
            x = self._dcnn_layer(
                x,
                self._n_filters_encoder[i],
                self._dilation_rate_encoder[i],
                _activation=self._activation_encoder[i],
                _kernel_size=self._kernel_size_encoder[i],
            )

        if not self.temporal_latent_space:
            shape_before_flatten = x.shape[1:]
            x = tf.keras.layers.Flatten()(x)
            output_layer = tf.keras.layers.Dense(self.latent_space_dim)(x)

        elif self.temporal_latent_space:
            output_layer = tf.keras.layers.Conv1D(
                filters=self.latent_space_dim,
                kernel_size=1,
            )(x)

        encoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        if self.temporal_latent_space:
            input_layer_decoder = tf.keras.layers.Input(x.shape[1:])
        elif not self.temporal_latent_space:
            input_layer_decoder = tf.keras.layers.Input((self.latent_space_dim,))
            dense_layer = tf.keras.layers.Dense(units=np.prod(shape_before_flatten))(
                input_layer_decoder
            )

            reshape_layer = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(
                dense_layer
            )
            input_layer_decoder = reshape_layer

        y = input_layer_decoder

        for i in range(0, self.n_layers):
            y = self._dcnn_layer_decoder(
                y,
                self._n_filters_encoder[::-1][i],
                self._dilation_rate_encoder[::-1][i],
                _activation=self._activation_encoder[::-1][i],
                _kernel_size=self._kernel_size_encoder[::-1][i],
            )

        last_layer = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)(y)
        decoder = tf.keras.Model(inputs=input_layer_decoder, outputs=last_layer)

        return encoder, decoder

    def _dcnn_layer(
        self, _inputs, _num_filters, _dilation_rate, _activation, _kernel_size
    ):
        import tensorflow as tf

        _add = tf.keras.layers.Conv1D(_num_filters, kernel_size=1)(_inputs)
        x = tf.keras.layers.Conv1D(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding="causal",
            kernel_regularizer="l2",
            activation=_activation,
        )(_inputs)
        x = tf.keras.layers.Conv1D(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding="causal",
            kernel_regularizer="l2",
            activation=_activation,
        )(x)
        output = tf.keras.layers.Add()([x, _add])
        return output

    def _dcnn_layer_decoder(
        self, _inputs, _num_filters, _dilation_rate, _activation, _kernel_size
    ):
        import tensorflow as tf

        _add = tf.keras.layers.Conv1DTranspose(_num_filters, kernel_size=1)(_inputs)
        x = tf.keras.layers.Conv1DTranspose(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding="same",
            kernel_regularizer="l2",
            activation=_activation,
        )(_inputs)
        x = tf.keras.layers.Conv1DTranspose(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding="same",
            kernel_regularizer="l2",
            activation=_activation,
        )(x)
        output = tf.keras.layers.Add()([x, _add])
        return output
