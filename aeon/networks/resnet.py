# -*- coding: utf-8 -*-
"""Residual Network (ResNet) (minus the final output layer)."""

__author__ = ["James Large", "Withington", "nilesh05apr", "hadifawaz1999"]

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies


class ResNetNetwork(BaseDeepNetwork):
    """
    Establish the network structure for a ResNet.

    Adapted from the implementations used in [1]

    Parameters
    ----------
        n_residual_blocks               : int, default = 3,
            the number of residual blocks of ResNet's model
        n_conv_per_residual_block       : int, default = 3,
            the number of convolution blocks in each residual block
        n_filters                       : int or list of int, default = [128, 64, 64],
            the number of convolution filters for all the convolution layers in the same
            residual block, if not a list, the same number of filters is used in all
            convolutions of all residual blocks.
        kernel_size                    : int or list of int, default = [8, 5, 3],
            the kernel size of all the convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        strides                         : int or list of int, default = 1,
            the strides of convolution kernels in each of the
            convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        dilation_rate                   : int or list of int, default = 1,
            the dilation rate of the convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        padding                         : str or list of str, default = 'padding',
            the type of padding used in the convolution layers
            in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        activation                      : str or list of str, default = 'relu',
            keras activation used in the convolution layers
            in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        use_bias                        : bool or list of bool, default = True,
            condition on wether or not to use bias values in
            the convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        random_state                    : int, optional (default = 0)
            The random seed to use random activities.

    Notes
    -----
    Adpated from the implementation source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    References
    ----------
    .. [1] H. Fawaz, G. B. Lanckriet, F. Petitjean, and L. Idoumghar,

    Network originally defined in:

    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }

    """

    _tags = {"python_dependencies": ["tensorflow", "keras-self-attention"]}

    def __init__(
        self,
        n_residual_blocks=3,
        n_conv_per_residual_block=3,
        n_filters=None,
        kernel_size=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        super(ResNetNetwork, self).__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block
        self.random_state = random_state

    def _shortcut_layer(
        self, input_tensor, output_tensor, padding="same", use_bias=True
    ):
        import tensorflow as tf

        n_out_filters = int(output_tensor.shape[-1])

        shortcut_layer = tf.keras.layers.Conv1D(
            filters=n_out_filters, kernel_size=1, padding=padding, use_bias=use_bias
        )(input_tensor)
        shortcut_layer = tf.keras.layers.BatchNormalization()(shortcut_layer)

        return tf.keras.layers.Add()([output_tensor, shortcut_layer])

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        input_layer : keras.layers.Input
            The input layer of the network.
        output_layer : keras.layers.Layer
            The output layer of the network.
        """
        import tensorflow as tf

        self._n_filters_ = [64, 128, 128] if self.n_filters is None else self.n_filters
        self._kernel_size_ = [8, 5, 3] if self.kernel_size is None else self.kernel_size

        if isinstance(self._n_filters_, list):
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_residual_blocks

        if isinstance(self._kernel_size_, list):
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_conv_per_residual_block

        if isinstance(self.strides, list):
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_conv_per_residual_block

        if isinstance(self.dilation_rate, list):
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_conv_per_residual_block

        if isinstance(self.padding, list):
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_conv_per_residual_block

        if isinstance(self.activation, list):
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_conv_per_residual_block

        if isinstance(self.use_bias, list):
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_conv_per_residual_block

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

        for d in range(self.n_residual_blocks):
            input_block_tensor = x

            for c in range(self.n_conv_per_residual_block):
                conv = tf.keras.layers.Conv1D(
                    filters=self._n_filters[d],
                    kernel_size=self._kernel_size[c],
                    strides=self._kernel_size[c],
                    padding=self._padding[c],
                    dilation_rate=self._dilation_rate[c],
                )(x)
                conv = tf.keras.layers.BatchNormalization()(conv)

                if c == self.n_conv_per_residual_block - 1:
                    conv = self._shortcut_layer(
                        input_tensor=input_block_tensor, output_tensor=conv
                    )

                conv = tf.keras.layers.Activation(activation=self._activation[c])(conv)

                x = conv

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv)

        return input_layer, gap_layer
