"""Auto-Encoder based on Dilated Convolutional Nerual Networks (DCNN) Model."""

__maintainer__ = []

from aeon.networks.base import BaseDeepNetwork


class AEDCNNNetwork(BaseDeepNetwork):
    """Establish the network structure for a DCNN-Model.

    Dilated Convolutional Neural Network based Model
    for low-rank embeddings.

    Parameters
    ----------
    latent_space_dim: int, default=128
        Dimension of the models's latent space.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    num_layers_encoder: int, default=4
        Number of convolution layers of the encoder.
    num_layers_decoder: int, default=4
        Number of convolution layers of the decoder.
    kernel_size_encoder: int, default=3
        Size of the 1D Convolutional Kernel of the encoder.
    kernel_size_decoder: int, default=3
        Size of the 1D Convolutional Kernel of the decoder.
    activation_encoder: str, default="relu"
        The activation function used by convolution layers of the encoder.
    activation_decoder: str, default="relu"
        The activation function used by convolution layers of the decoder.
    num_filters_encoder: int, default=None
        Number of filters used in convolution layers of the encoder.
    num_filters_decoder: int, default=None
        Number of filters used in convolution layers of the decoder.
    dilation_rate_encoder: list, default=None
        The dilation rate for convolution of the encoder.
    dilation_rate_decoder: list, default=None
        The dilation rate for convolution of the decoder.

    References
    ----------
    .. [1] Network originally defined in:
    @article{franceschi2019unsupervised,
      title={Unsupervised scalable representation learning for multivariate time
        series},
      author={Franceschi, Jean-Yves and Dieuleveut, Aymeric and Jaggi, Martin},
      journal={Advances in neural information processing systems},
      volume={32},
      year={2019}
    }
    """

    _tags = {"auto-encoder": True}

    def __init__(
        self,
        latent_space_dim=128,
        temporal_latent_space=False,
        num_layers_encoder=4,
        num_layers_decoder=4,
        kernel_size_encoder=3,
        kernel_size_decoder=None,
        activation_encoder="relu",
        activation_decoder="relu",
        num_filters_encoder=None,
        num_filters_decoder=None,
        dilation_rate_encoder=None,
        dilation_rate_decoder=None,
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.kernel_size_encoder = kernel_size_encoder
        self.kernel_size_decoder = kernel_size_decoder
        self.num_filters_encoder = num_filters_encoder
        self.num_filters_decoder = num_filters_decoder
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.dilation_rate_encoder = dilation_rate_encoder
        self.dilation_rate_decoder = dilation_rate_decoder
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
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

        if self.num_filters_encoder is None:
            self._num_filters_encoder = [
                32 * i for i in range(1, self.num_layers_encoder + 1)
            ]
        elif isinstance(self.num_filters_encoder, list):
            self._num_filters_encoder = self.num_filters_encoder
            assert len(self.num_filters_encoder) == self.num_layers_encoder

        if self.dilation_rate_encoder is None:
            self._dilation_rate_encoder = [
                2**layer_num for layer_num in range(1, self.num_layers_encoder + 1)
            ]
        elif isinstance(self.dilation_rate_encoder, int):
            self._dilation_rate_encoder = [
                self._dilation_rate_encoder for _ in range(self.num_layers_encoder)
            ]
        else:
            self._dilation_rate_encoder = self.dilation_rate_encoder
            assert isinstance(self.dilation_rate_encoder, list)
            assert len(self.dilation_rate_encoder) == self.num_layers_encoder

        if isinstance(self.kernel_size_encoder, int):
            self._kernel_size_encoder = [
                self.kernel_size_encoder for _ in range(self.num_layers_encoder)
            ]
        elif isinstance(self.kernel_size_encoder, list):
            self._kernel_size_encoder = self.kernel_size_encoder
            assert len(self.kernel_size_encoder) == self.num_layers_encoder

        if isinstance(self.activation_encoder, str):
            self._activation_encoder = [
                self.activation_encoder for _ in range(self.num_layers_encoder)
            ]
        elif isinstance(self.activation_encoder, list):
            self._activation_encoder = self.activation_encoder
            assert len(self._activation_encoder) == self.num_layers_encoder

        if self.activation_decoder is None:
            self._activation_decoder = self._activation_encoder[::-1]
        elif isinstance(self.activation_decoder, str):
            self._activation_decoder = [
                self.activation_decoder for _ in range(self.num_layers_decoder)
            ]
        else:
            self._activation_decoder = self.activation_decoder
            assert isinstance(self.activation_decoder, list)
            assert len(self.activation_decoder) == self.num_layers_decoder

        if self.dilation_rate_decoder is None:
            self._dilation_rate_decoder = self._dilation_rate_encoder[::-1]
        elif isinstance(self.dilation_rate_decoder, int):
            self._dilation_rate_decoder = [
                self._dilation_rate_decoder for _ in range(self.num_layers_decoder)
            ]
        else:
            self._dilation_rate_decoder = self._dilation_rate_decoder
            assert isinstance(self._dilation_rate_decoder, list)
            assert len(self._dilation_rate_decoder) == self.num_layers_decoder

        if self.kernel_size_decoder is None:
            self._kernel_size_decoder = self._kernel_size_encoder[::-1]
        elif isinstance(self.kernel_size_decoder, int):
            self._kernel_size_decoder = [
                self.kernel_size_decoder for _ in range(self.num_layers_decoder)
            ]
        else:
            self._kernel_size_decoder = self.kernel_size_decoder
            assert isinstance(self.kernel_size_decoder, list)
            assert len(self.kernel_size_decoder) == self.num_layers_decoder

        if self.dilation_rate_decoder is None:
            self._dilation_rate_decoder = self._dilation_rate_encoder
        elif isinstance(self.dilation_rate_decoder, int):
            self._dilation_rate_decoder = [
                self._dilation_rate_decoder for _ in range(self.num_layers_decoder)
            ]
        else:
            self._dilation_rate_decoder = self.dilation_rate_decoder
            assert isinstance(self.dilation_rate_decoder, list)
            assert len(self.dilation_rate_decoder) == self.num_layers_decoder

        if self.num_filters_decoder is None:
            self._num_filters_decoder = self._num_filters_encoder
        elif isinstance(self.num_filters_decoder, list):
            self._num_filters_decoder = self.num_filters_decoder
            assert len(self.num_filters_encoder) == self.num_layers_decoder

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(0, self.num_layers_encoder):
            x = self._dcnn_layer(
                x,
                self._num_filters_encoder[i],
                self._dilation_rate_encoder[i],
                _activation=self._activation_encoder[i],
                _kernel_size=self._kernel_size_encoder[i],
            )

        if not self.temporal_latent_space:
            x = tf.keras.layers.GlobalMaxPool1D()(x)
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

        y = input_layer_decoder

        for i in range(0, self.num_layers_decoder):
            y = self._dcnn_layer(
                y,
                self._num_filters_decoder[i],
                self._dilation_rate_decoder[i],
                _activation=self._activation_decoder[i],
                _kernel_size=self._kernel_size_decoder[i],
            )

        decoder = tf.keras.Model(inputs=input_layer_decoder, outputs=y)

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
