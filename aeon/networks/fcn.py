# -*- coding: utf-8 -*-
"""Fully Connected Neural Network (FCN) (minus the final output layer)."""

__author__ = ["James-Large", "AurumnPegasus", "hadifawaz1999"]

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class FCNNetwork(BaseDeepNetwork):
    """
    Establish the network structure for a FCN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    n_layers : int, default = 3
        number of convolution layers
    n_filters : int or list of int, default = [128,256,128]
        number of filters used in convolution layers
    kernel_size : int or list of int, default = [8,5,3]
        size of convolution kernel
    dilation_rate : int or list of int, default = 1
        the dilation rate for convolution
    strides : int or list of int, default = 1
        the strides of the convolution filter
    padding : str or list of str, default = "same"
        the type of padding used for convolution
    activation : str or list of str, default = "relu"
        activation used after the convolution
    use_bias : bool or list of bool, default = True
        whether or not ot use bias in convolution
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

        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.random_state = random_state

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

        self._n_filters_ = [128, 256, 128] if self.n_filters is None else self.n_filters
        self._kernel_size_ = [8, 5, 3] if self.kernel_size is None else self.kernel_size

        if isinstance(self._n_filters_, list):
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_layers

        if isinstance(self._kernel_size_, list):
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_layers

        if isinstance(self.dilation_rate, list):
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_layers

        if isinstance(self.strides, list):
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_layers

        if isinstance(self.padding, list):
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_layers

        if isinstance(self.activation, list):
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_layers

        if isinstance(self.use_bias, list):
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

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
            conv = tf.keras.layers.Activation(activation=self._activation[i])(conv)

            x = conv

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv)

        return input_layer, gap_layer
