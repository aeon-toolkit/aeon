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
        Whether or not ot use bias in convolution.

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
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
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
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        super().__init__()

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

        self._n_filters_ = [128, 256, 128] if self.n_filters is None else self.n_filters
        self._kernel_size_ = [8, 5, 3] if self.kernel_size is None else self.kernel_size

        if isinstance(self._n_filters_, list):
            if len(self._n_filters_) != self.n_layers:
                raise ValueError(
                    f"Number of filters {len(self._n_filters_)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_layers

        if isinstance(self._kernel_size_, list):
            if len(self._kernel_size_) != self.n_layers:
                raise ValueError(
                    f"Number of kernels {len(self._kernel_size_)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._kernel_size = self.kernel_size
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_layers

        if isinstance(self.dilation_rate, list):
            if len(self.dilation_rate) != self.n_layers:
                raise ValueError(
                    f"Number of dilations {len(self.dilation_rate)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_layers

        if isinstance(self.strides, list):
            if len(self.strides) != self.n_layers:
                raise ValueError(
                    f"Number of strides {len(self.strides)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_layers

        if isinstance(self.padding, list):
            if len(self.padding) != self.n_layers:
                raise ValueError(
                    f"Number of paddings {len(self.padding)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_layers

        if isinstance(self.activation, list):
            if len(self.activation) != self.n_layers:
                raise ValueError(
                    f"Number of activations {len(self.activation)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_layers

        if isinstance(self.use_bias, list):
            if len(self.use_bias) != self.n_layers:
                raise ValueError(
                    f"Number of biases {len(self.use_bias)} should be"
                    f" the same as number of layers but is"
                    f" not: {self.n_layers}"
                )
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_layers

        input_layer_encoder = tf.keras.layers.Input(input_shape)

        x = input_layer_encoder

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

            dense_layer = tf.keras.layers.Dense(units=np.prod(shape_before_flattent))(
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
