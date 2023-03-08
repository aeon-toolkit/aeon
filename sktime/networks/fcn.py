# -*- coding: utf-8 -*-
"""Fully Connected Neural Network (FCN) (minus the final output layer)."""

__author__ = ["James-Large", "AurumnPegasus", "hadifawaz1999"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class FCNNetwork(BaseDeepNetwork):
    """
    Establish the network structure for a FCN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    random_state    : int, default = 0
        seed to any needed random actions

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

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_layers=3,
        n_filters=None,
        kernel_size=None,
        dilation_rate=1,
        strides=1,
        padding="same",
        activation="relu",
        use_bias=True,
        random_state=0,
    ):
        super(FCNNetwork, self).__init__()
        _check_dl_dependencies(severity="error")

        self.n_filters = [128, 256, 128] if n_filters is None else n_filters
        self.kernel_size = [8, 5, 3] if kernel_size is None else kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        self.n_layers = n_layers

        if isinstance(self.n_filters, list):
            self.n_filters = self.n_filters
        else:
            self.n_filters = [self.n_filters] * self.n_layers

        if isinstance(self.kernel_size, list):
            self.kernel_size = self.kernel_size
        else:
            self.kernel_size = [self.kernel_size] * self.n_layers

        if isinstance(self.dilation_rate, list):
            self.dilation_rate = self.dilation_rate
        else:
            self.dilation_rate = [self.dilation_rate] * self.n_layers

        if isinstance(self.strides, list):
            self.strides = self.strides
        else:
            self.strides = [self.strides] * self.n_layers

        if isinstance(self.padding, list):
            self.padding = self.padding
        else:
            self.padding = [self.padding] * self.n_layers

        if isinstance(self.activation, list):
            self.activation = self.activation
        else:
            self.activation = [self.activation] * self.n_layers

        if isinstance(self.use_bias, list):
            self.use_bias = self.use_bias
        else:
            self.use_bias = [self.use_bias] * self.n_layers

        self.random_state = random_state

        assert len(self.kernel_size) == self.n_layers
        assert len(self.n_filters) == self.n_layers
        assert len(self.padding) == self.n_layers
        assert len(self.strides) == self.n_layers
        assert len(self.activation) == self.n_layers
        assert len(self.dilation_rate) == self.n_layers
        assert len(self.use_bias) == self.n_layers

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

        for i in range(self.n_layers):
            conv = tf.keras.layers.Conv1D(
                filters=self.n_filters[i],
                kernel_size=self.kernel_size[i],
                strides=self.strides[i],
                dilation_rate=self.dilation_rate[i],
                padding=self.padding[i],
                use_bias=self.use_bias[i],
            )(x)

            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.Activation(activation=self.activation[i])(conv)

            x = conv

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv)

        return input_layer, gap_layer
