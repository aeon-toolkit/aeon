# -*- coding: utf-8 -*-
"""Residual Network (ResNet) (minus the final output layer)."""

__author__ = ["James Large", "Withington", "nilesh05apr", "hadifawaz1999"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies


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
        kernel_sizes                    : int or list of int, default = [8, 5, 3],
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
        kernel_sizes=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        super(ResNetNetwork, self).__init__()

        n_filters = [64, 128, 128] if n_filters is None else n_filters
        kernel_sizes = [8, 5, 3] if kernel_sizes is None else kernel_sizes

        self.n_residual_blocks = n_residual_blocks
        self.n_conv_per_residual_block = n_conv_per_residual_block

        if isinstance(n_filters, list):
            self.n_filters = n_filters
        else:
            self.n_filters = [n_filters] * self.n_residual_blocks

        if isinstance(kernel_sizes, list):
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * self.n_conv_per_residual_block

        if isinstance(strides, list):
            self.strides = strides
        else:
            self.strides = [strides] * self.n_conv_per_residual_block

        if isinstance(dilation_rate, list):
            self.dilation_rate = dilation_rate
        else:
            self.dilation_rate = [dilation_rate] * self.n_conv_per_residual_block

        if isinstance(padding, list):
            self.padding = padding
        else:
            self.padding = [padding] * self.n_conv_per_residual_block

        if isinstance(activation, list):
            self.activation = activation
        else:
            self.activation = [activation] * self.n_conv_per_residual_block

        if isinstance(use_bias, list):
            self.use_bias = use_bias
        else:
            self.use_bias = [use_bias] * self.n_conv_per_residual_block

        assert len(self.n_filters) == self.n_residual_blocks
        assert len(self.kernel_sizes) == self.n_conv_per_residual_block
        assert len(self.strides) == self.n_conv_per_residual_block
        assert len(self.padding) == self.n_conv_per_residual_block
        assert len(self.dilation_rate) == self.n_conv_per_residual_block
        assert len(self.activation) == self.n_conv_per_residual_block
        assert len(self.use_bias) == self.n_conv_per_residual_block

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

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

        for d in range(self.n_residual_blocks):
            input_block_tensor = x

            for c in range(self.n_conv_per_residual_block):
                conv = tf.keras.layers.Conv1D(
                    filters=self.n_filters[d],
                    kernel_size=self.kernel_sizes[c],
                    strides=self.kernel_sizes[c],
                    padding=self.padding[c],
                    dilation_rate=self.dilation_rate[c],
                )(x)
                conv = tf.keras.layers.BatchNormalization()(conv)

                if c == self.n_conv_per_residual_block - 1:
                    conv = self._shortcut_layer(
                        input_tensor=input_block_tensor, output_tensor=conv
                    )

                conv = tf.keras.layers.Activation(activation=self.activation[c])(conv)

                x = conv

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv)

        return input_layer, gap_layer
