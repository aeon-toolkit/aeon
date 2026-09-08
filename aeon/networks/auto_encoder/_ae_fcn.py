"""Auto-Encoder using Fully Convolutional Network (FCN)."""

__maintainer__ = ["hadifawaz1999"]

import numpy as np

from aeon.networks.base import BaseDeepLearningNetwork


class AEFCNNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a AE-FCN.

    Auto-Encoder based Fully Convolutional Netwwork (AE-FCN),
    adapted from the implementation used in [1]_.

    Parameters
    ----------
    latent_space_dim : int, default = 128
        Dimension of the auto-encoder's latent space.
    temporal_latent_space : bool, default = False
        Flag to choose whether the latent space is an MTS or Euclidean space.
    n_layers : int, default = 3
        Number of convolution layers.
    n_filters : int or list of int, default = [128,256,128]
        Number of filters used in convolution layers.
    kernel_size : int or list of int, default = [8,5,3]
        Size of convolution kernel.
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution.
    strides : int or list of int, default = 1
        The strides of the convolution filter.
    padding : str or list of str, default = "same"
        The type of padding used for convolution.
    activation : str or list of str, default = "relu"
        Activation used after the convolution.
    use_bias : bool or list of bool, default = True
        Whether or not to use bias in convolution.

    Notes
    -----
    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    References
    ----------
    .. [1] Network originally defined in:
    @inproceedings{wang2017time,
      title={Time series classification from scratch with deep neural networks:
       A strong baseline},
      author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
      booktitle={2017 International joint conference on neural networks
      (IJCNN)},
      pages={1578--1585},
      year={2017},
      organization={IEEE}
    }
    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        latent_space_dim=128,
        temporal_latent_space=False,
        n_layers=3,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=True,
    ):
        self.latent_space_dim = latent_space_dim
        self.temporal_latent_space = temporal_latent_space
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        super().__init__()

    def _check_params(self):
        n = self.n_layers
        self._n_filters = BaseDeepLearningNetwork._check_layer_param(
            n, self.n_filters, "filters", default=[128, 256, 128]
        )
        self._kernel_size = BaseDeepLearningNetwork._check_layer_param(
            n, self.kernel_size, "kernel size", default=[8, 5, 3]
        )
        self._dilation_rate = BaseDeepLearningNetwork._check_layer_param(
            n, self.dilation_rate, "dilation rate", default=1
        )
        self._strides = BaseDeepLearningNetwork._check_layer_param(
            n, self.strides, "strides", default=1
        )
        self._padding = BaseDeepLearningNetwork._check_layer_param(
            n, self.padding, "padding", default="same"
        )
        self._activation = BaseDeepLearningNetwork._check_layer_param(
            n, self.activation, "activations", allow_none=True
        )
        self._use_bias = BaseDeepLearningNetwork._check_layer_param(
            n, self.use_bias, "use bias", default=True
        )

    def build_base_graph(self, x):
        self._check_params()
        return x

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
        import tensorflow as tf

        input_layer_encoder = tf.keras.layers.Input(input_shape)
        x = input_layer_encoder
        x = self.build_base_graph(x)

        for i in range(self.n_layers):
            conv = tf.keras.layers.Conv1D(
                filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                strides=self._strides[i],
                dilation_rate=self._dilation_rate[i],
                padding=self._padding[i],
                use_bias=self._use_bias[i],
            )(x)

            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.Activation(
                activation=self._activation[i], name=f"__act_encoder_block{i}"
            )(conv)

            x = conv

        if not self.temporal_latent_space:
            shape_before_flattent = x.shape[1:]
            flatten_layer = tf.keras.layers.Flatten()(x)
            latent_space = tf.keras.layers.Dense(units=self.latent_space_dim)(
                flatten_layer
            )
        else:
            latent_space = tf.keras.layers.Conv1D(
                filters=self.latent_space_dim,
                kernel_size=1,
                strides=self._strides[-1],
                padding=self._padding[-1],
                dilation_rate=self._dilation_rate[-1],
                use_bias=self._use_bias[-1],
            )(x)

        encoder = tf.keras.models.Model(
            inputs=input_layer_encoder, outputs=latent_space, name="encoder"
        )

        if not self.temporal_latent_space:
            input_layer_decoder = tf.keras.layers.Input((self.latent_space_dim,))

            # Cast to int to avoid Keras rejecting numpy scalar types
            decoder_units = int(np.prod(shape_before_flattent))
            dense_layer = tf.keras.layers.Dense(units=decoder_units)(
                input_layer_decoder
            )

            reshape_layer = tf.keras.layers.Reshape(target_shape=shape_before_flattent)(
                dense_layer
            )
            x = reshape_layer
        else:
            input_layer_decoder = tf.keras.layers.Input(latent_space.shape[1:])

            x = input_layer_decoder

        for i in range(self.n_layers)[::-1]:
            conv = tf.keras.layers.Conv1DTranspose(
                filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                strides=self._strides[i],
                dilation_rate=self._dilation_rate[i],
                padding=self._padding[i],
                use_bias=self._use_bias[i],
            )(x)

            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.Activation(
                activation=self._activation[i], name=f"__act_decoder_block{i}"
            )(conv)

            x = conv

        last_projection_layer = tf.keras.layers.Conv1DTranspose(
            filters=input_shape[-1],
            kernel_size=1,
            padding=self._padding[0],
            strides=self._strides[0],
            dilation_rate=self._dilation_rate[0],
            use_bias=self._use_bias[0],
        )(x)

        decoder = tf.keras.models.Model(
            inputs=input_layer_decoder, outputs=last_projection_layer, name="decoder"
        )

        return encoder, decoder
