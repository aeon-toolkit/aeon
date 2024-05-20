"""Encoder using Dilated Convolutional Nerual Networks (DCNN)."""

__maintainer__ = []

import tensorflow as tf

from aeon.networks.base import BaseDeepNetwork


class DCNNEncoderNetwork(BaseDeepNetwork):
    """Establish the network structure for a DCNN-Encoder.

    Dilated Convolutional Neural Network based Encoder
    for low-rank embeddings.

    Parameters
    ----------
    latent_space_dim: int, default=128
        Dimension of the encoder's latent space.
    num_layers: int, default=4
        Number of convolution layers.
    kernel_size: int, default=3
        Size of the 1D Convolutional Kernel.
    num_filters: int, default=None
        Number of filters used in convolution layers.
    dilation_rate: list, default=None
        The dilation rate for convolution.
    activation: str, default="relu"
        The activation function used by convolution layers.

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
        self.model_depth = num_layers
        self.dilation_rate = dilation_rate
        self.activation = activation

        if self.num_filters is None:
            self.num_filters = [32 * i for i in range(1, self.model_depth + 1)]

        if self.dilation_rate is None:
            self.dilation_rate = [
                2**layer_num for layer_num in range(1, self.model_depth + 1)
            ]
        else:
            assert isinstance(self.dilation_rate, list)
            assert len(self.dilation_rate) == self.model_depth

    def build_network(self, input_shape):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        encoder : a keras Model.
        """

        def DCNNLayer(inputs, num_filters, dilation_rate, activation):
            _add = tf.keras.layers.Conv1D(num_filters, kernel_size=1)(inputs)
            x = tf.keras.layers.Conv1D(
                num_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                padding="causal",
                kernel_regularizer="l2",
                activation=activation,
            )(inputs)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Conv1D(
                num_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                padding="causal",
                kernel_regularizer="l2",
                activation=activation,
            )(x)
            x = tf.keras.layers.LeakyReLU()(x)
            output = tf.keras.layers.Add()([x, _add])
            return output

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(0, self.model_depth):
            x = DCNNLayer(
                x,
                self.num_filters[i],
                self.dilation_rate[i],
                activation=self.activation,
            )
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        output_layer = tf.keras.layers.Dense(self.latent_space_dim)(x)

        encoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return encoder
