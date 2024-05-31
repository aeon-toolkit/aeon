"""Dilated Convolutional Nerual Networks (DCNN) Model."""

__maintainer__ = []

from aeon.networks.base import BaseDeepNetwork


class DCNNNetwork(BaseDeepNetwork):
    """Establish the network structure for a DCNN-Model.

    Dilated Convolutional Neural Network based Model
    for low-rank embeddings.

    Parameters
    ----------
    latent_space_dim: int, default=128
        Dimension of the models's latent space.
    num_layers: int, default=4
        Number of convolution layers.
    kernel_size: int, default=3
        Size of the 1D Convolutional Kernel.
    activation: str, default="relu"
        The activation function used by convolution layers.
    num_filters: int, default=None
        Number of filters used in convolution layers.
    dilation_rate: list, default=None
        The dilation rate for convolution.

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

    def __init__(
        self,
        latent_space_dim=128,
        num_layers=4,
        kernel_size=3,
        activation="relu",
        num_filters=None,
        dilation_rate=None,
    ):
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dilation_rate = dilation_rate
        self.activation = activation

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

        if self.num_filters is None:
            self._num_filters = [32 * i for i in range(1, self.num_layers + 1)]
        elif isinstance(self.num_filters, list):
            self._num_filters = self.num_filters
            assert len(self.num_filters) == self.num_layers

        if self.dilation_rate is None:
            self._dilation_rate = [
                2**layer_num for layer_num in range(1, self.num_layers + 1)
            ]
        else:
            self._dilation_rate = self.dilation_rate
            assert isinstance(self.dilation_rate, list)
            assert len(self.dilation_rate) == self.num_layers

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size for _ in range(self.num_layers)]
        elif isinstance(self.kernel_size, list):
            self._kernel_size = self.kernel_size
            assert len(self.kernel_size) == self.num_layers

        if isinstance(self.activation, str):
            self.activation = [self.activation for _ in range(self.num_layers)]
        elif isinstance(self.activation, list):
            self._activation = self.activation
            assert len(self.activation) == self.num_layers

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(0, self.num_layers):
            x = self._dcnn_layer(
                x,
                self._num_filters[i],
                self._dilation_rate[i],
                _activation=self._activation[i],
                _kernel_size=self._kernel_size[i],
            )
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        output_layer = tf.keras.layers.Dense(self.latent_space_dim)(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return model

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
