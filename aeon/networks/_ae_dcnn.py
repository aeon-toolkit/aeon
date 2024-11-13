"""Auto-Encoder based on Dilated Convolutional Nerual Networks (DCNN) Model."""

__maintainer__ = ["aadya940", "hadifawaz1999"]

import warnings

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
        Size of the 1D Convolutional Kernel of the encoder. Defaults to a
        list of length `n_layers` with `kernel_size` value.
    activation: Union[str, List[str]], default="relu"
        The activation function used by convolution layers of the encoder.
        Defaults to a list of "relu" for `n_layers` elements.
    n_filters: Union[int, List[int]], default=None
        Number of filters used in convolution layers of the encoder. Defaults
        to a list of multiples of `32` for `n_layers` elements.
    dilation_rate: Union[int, List[int]], default=1
        The dilation rate for convolution of the encoder. Defaults to a list
        of powers of `2` for `n_layers` elements. `dilation_rate` greater than
        `1` is not supported on `Conv1DTranspose` for some devices/OS.
    padding_encoder: Union[str, List[str]], default="same"
        The padding string for the encoder layers. Defaults to a list of "same"
        for `n_layers` elements. Valid strings are "causal", "valid", "same" or
        any other Keras compatible string.
    padding_decoder: Union[str, List[str]], default="same"
        The padding string for the decoder layers. Defaults to a list of "same"
        for `n_layers` elements.

    References
    ----------
    .. [1] Franceschi, J. Y., Dieuleveut, A., & Jaggi, M. (2019). Unsupervised
    scalable representation learning for multivariate time series. Advances in
    neural information processing systems, 32.

    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
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
        dilation_rate=1,
        padding_encoder="same",
        padding_decoder="same",
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.temporal_latent_space = temporal_latent_space
        self.padding_encoder = padding_encoder
        self.padding_decoder = padding_decoder

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
        elif isinstance(self.n_filters, int):
            self._n_filters_encoder = [self.n_filters for _ in range(self.n_layers)]
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

        if self.activation is None:
            self._activation_encoder = ["relu" for _ in range(self.n_layers)]
        elif isinstance(self.activation, str):
            self._activation_encoder = [self.activation for _ in range(self.n_layers)]
        elif isinstance(self.activation, list):
            self._activation_encoder = self.activation
            assert len(self._activation_encoder) == self.n_layers

        if self.padding_encoder is None:
            self._padding_encoder = ["same" for _ in range(self.n_layers)]
        elif isinstance(self.padding_encoder, str):
            self._padding_encoder = [self.padding_encoder for _ in range(self.n_layers)]
        elif isinstance(self.padding_encoder, list):
            self._padding_encoder = self.padding_encoder
            assert len(self._padding_encoder) == self.n_layers

        if self.padding_decoder is None:
            self._padding_decoder = ["same" for _ in range(self.n_layers)]
        elif isinstance(self.padding_decoder, str):
            self._padding_decoder = [self.padding_decoder for _ in range(self.n_layers)]
        elif isinstance(self.padding_decoder, list):
            self._padding_decoder = self.padding_decoder
            assert len(self._padding_decoder) == self.n_layers

        if self.dilation_rate == 1 or np.all(
            np.array(self._dilation_rate_encoder) == 1
        ):
            warnings.warn(
                """Currently, the dilation rate has been set to `1` which is
            different from the original paper of the `AEDCNNNetwork` due to CPU
            Implementation issues with `tensorflow.keras.layers.Conv1DTranspose`
            & `dilation_rate` > 1 on some Hardwares & OS combinations. You
            can use the dilation rates as specified in the paper by passing
            `dilation_rate=None` to the Network/Clusterer.""",
                UserWarning,
                stacklevel=2,
            )

        if np.any(np.array(self._dilation_rate_encoder) > 1):
            warnings.warn(
                """Current network configuration contains `dilation_rate`
                more than 1, which is not supported by
                `tensorflow.keras.layers.Conv1DTranspose` layer for certain
                hardware architectures and/or Operating Systems.""",
                UserWarning,
                stacklevel=2,
            )

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(0, self.n_layers):
            x = self._dcnn_layer(
                x,
                self._n_filters_encoder[i],
                self._dilation_rate_encoder[i],
                _activation=self._activation_encoder[i],
                _kernel_size=self._kernel_size_encoder[i],
                _padding_encoder=self._padding_encoder[i],
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
            temp = input_layer_decoder
        elif not self.temporal_latent_space:
            input_layer_decoder = tf.keras.layers.Input((self.latent_space_dim,))
            dense_layer = tf.keras.layers.Dense(units=np.prod(shape_before_flatten))(
                input_layer_decoder
            )

            reshape_layer = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(
                dense_layer
            )
            temp = reshape_layer

        y = temp

        for i in range(0, self.n_layers):
            y = self._dcnn_layer_decoder(
                y,
                self._n_filters_encoder[::-1][i],
                self._dilation_rate_encoder[::-1][i],
                _activation=self._activation_encoder[::-1][i],
                _kernel_size=self._kernel_size_encoder[::-1][i],
                _padding_decoder=self._padding_decoder[i],
            )

        last_layer = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)(y)
        decoder = tf.keras.Model(inputs=input_layer_decoder, outputs=last_layer)

        return encoder, decoder

    def _dcnn_layer(
        self,
        _inputs,
        _num_filters,
        _dilation_rate,
        _activation,
        _kernel_size,
        _padding_encoder,
    ):
        import tensorflow as tf

        _add = tf.keras.layers.Conv1D(_num_filters, kernel_size=1)(_inputs)
        x = tf.keras.layers.Conv1D(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding=_padding_encoder,
            kernel_regularizer="l2",
        )(_inputs)
        x = tf.keras.layers.Conv1D(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding=_padding_encoder,
            kernel_regularizer="l2",
        )(x)
        output = tf.keras.layers.Add()([x, _add])
        output = tf.keras.layers.Activation(_activation)(output)
        return output

    def _dcnn_layer_decoder(
        self,
        _inputs,
        _num_filters,
        _dilation_rate,
        _activation,
        _kernel_size,
        _padding_decoder,
    ):
        import tensorflow as tf

        _add = tf.keras.layers.Conv1DTranspose(_num_filters, kernel_size=1)(_inputs)
        x = tf.keras.layers.Conv1DTranspose(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding=_padding_decoder,
            kernel_regularizer="l2",
        )(_inputs)
        x = tf.keras.layers.Conv1DTranspose(
            _num_filters,
            kernel_size=_kernel_size,
            dilation_rate=_dilation_rate,
            padding=_padding_decoder,
            kernel_regularizer="l2",
        )(x)
        output = tf.keras.layers.Add()([x, _add])
        output = tf.keras.layers.Activation(_activation)(output)
        return output
