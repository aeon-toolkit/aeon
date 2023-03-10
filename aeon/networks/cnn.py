# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

__author__ = ["James-Large", "Withington", "TonyBagnall", "hadifawaz1999"]

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class CNNNetwork(BaseDeepNetwork):
    """Establish the network structure for a CNN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
        n_layers        : int, default = 2,
            the number of convolution layers in the network
        kernel_size    : int or list of int, default = 7,
            kernel size of convolution layers, if not a list, the same kernel size
            is used for all layer, len(list) should be n_layers
        n_filters       : int or list of int, default = [6, 12],
            number of filters for each convolution layer,
            if not a list, the same n_filters
            is used in all layers.
        avg_pool_size   : int or list of int, default = 3,
            the size of the average pooling layer, if not a
            list, the same max pooling size is used
            for all convolution layer
        activation      : str or list of str, default = "sigmoid",
            keras activation function used in the model for
            each layer, if not a list, the same
            activation is used for all layers
        padding         : str or list of str, default = 'valid',
            the method of padding in convolution layers, if
            not a list, the same padding used
            for all convolution layers
        strides         : int or list of int, default = 1,
            the strides of kernels in the convolution and
            max pooling layers, if not a list,
            the same strides are used for all layers
        dilation_rate   : int or list of int, default = 1,
            the dilation rate of the convolution layers,
            if not a list, the same dilation rate
            is used all over the network
        use_bias        : bool or list of bool, default = True,
            condition on wether or not to use bias values
            for convolution layers, if not
            a list, the same condition is used for all layers
        random_state    : int, default = 0
            seed to any needed random actions

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_layers=2,
        kernel_size=7,
        n_filters=None,
        avg_pool_size=3,
        activation="sigmoid",
        padding="valid",
        strides=1,
        dilation_rate=1,
        use_bias=True,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        self.n_layers = n_layers
        self.random_state = random_state
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        super(CNNNetwork, self).__init__()

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        self._n_filters_ = [6, 12] if self.n_filters is None else self.n_filters

        if isinstance(self.kernel_size, list):
            self._kernel_size = self.kernel_size
        else:
            self._kernel_size = [self.kernel_size] * self.n_layers

        if isinstance(self._n_filters_, list):
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_layers

        if isinstance(self.avg_pool_size, list):
            self._avg_pool_size = self.avg_pool_size
        else:
            self._avg_pool_size = [self.avg_pool_size] * self.n_layers

        if isinstance(self.activation, list):
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_layers

        if isinstance(self.padding, list):
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_layers

        if isinstance(self.strides, list):
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_layers

        if isinstance(self.dilation_rate, list):
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_layers

        if isinstance(self.use_bias, list):
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_layers

        input_layer = tf.keras.layers.Input(input_shape)
        # sort this out, why hard coded to 60? [hadifawaz1999 answer:
        # It was proposed like that in the original paper, check code in
        # https://github.com/hfawaz/dl-4-tsc/blob/3ee62e16e118e4f5cfa86d01661846dfa75febfa/classifiers/cnn.py#L30]

        """

        When the UCR archive was published in 2015, it
        contained 85 datasets, each dataset had
        different characteristics such as the time series length.

        The smallest dataset (refering to time series length)
        was the ItalyPowerDemand dataset
        with a length of 24,
        the second smallest one was SyntheticControl with a length of 60.

        The authors in [1], added the padding condition on the
        length 60 because it was the second smallest
        dataset (refering to time series length), SyntheticControl.

        Padding helps in the case of short time series because
        the samples would contain a small amount
        of information, which leads to a loss of information
        when applying convolutions without padding.
        This is due to the fact to negleting the side
        parts of the time series.

        Note that when the UCR archive had new datasets,
        128 in 2018, more datasets of small length were added.
        For this reason, we keep the conditioning on 60
        as length of input time series to add the padding in
        the convolution.

        """

        if input_shape[0] < 60:
            self._padding = ["same"] * self.n_layers

        x = input_layer

        for i in range(self.n_layers):
            conv = tf.keras.layers.Conv1D(
                filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                strides=self._strides[i],
                padding=self._padding[i],
                dilation_rate=self._dilation_rate[i],
                activation=self._activation[i],
                use_bias=self._use_bias[i],
            )(x)

            conv = tf.keras.layers.AveragePooling1D(pool_size=self._avg_pool_size[i])(
                conv
            )

            x = conv

        flatten_layer = tf.keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
