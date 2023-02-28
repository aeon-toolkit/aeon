# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

__author__ = ["James-Large, Withington, TonyBagnall"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class CNNNetwork(BaseDeepNetwork):
    """Establish the network structure for a CNN.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_size     : int, default = 7
        specifying the length of the 1D convolution window
    avg_pool_size   : int, default = 3
        size of the average pooling windows
    n_conv_layers   : int, default = 2
        the number of convolutional plus average pooling layers
    filter_sizes    : array of int, shape = (nb_conv_layers)
    activation      : string, default = sigmoid
        keras activation function
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
        kernel_sizes=7,
        n_filters=[6, 12],
        avg_pool_size=3,
        activation="sigmoid",
        padding='valid',
        strides=1,
        dilation_rate=1,
        use_bias=True,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state

        if isinstance(kernel_sizes, list):
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * n_layers
        
        if isinstance(n_filters, list):
            self.n_filters = n_filters
        else:
            self.n_filters = [n_filters] * n_layers
        
        if isinstance(avg_pool_size, list):
            self.avg_pool_size = avg_pool_size
        else:
            self.avg_pool_size = [avg_pool_size] * n_layers
        
        if isinstance(activation, list):
            self.activation = activation
        else:
            self.activation = [activation] * n_layers
        
        if isinstance(padding, list):
            self.padding = padding
        else:
            self.padding = [padding] * n_layers
        
        if isinstance(strides, list):
            self.strides = strides
        else:
            self.strides = [strides] * n_layers
        
        if isinstance(dilation_rate, list):
            self.dilation_rate = dilation_rate
        else:
            self.dilation_rate = [dilation_rate] * n_layers
        
        if isinstance(use_bias, list):
            self.use_bias = use_bias
        else:
            self.use_bias = [use_bias] * n_layers
        
        self.n_layers = n_layers

        assert(len(self.kernel_sizes) == n_layers)
        assert(len(self.avg_pool_size) == n_layers)
        assert(len(self.strides) == n_layers)
        assert(len(self.dilation_rate) == n_layers)
        assert(len(self.padding) == n_layers)
        assert(len(self.activation) == n_layers)

        super(CNNNetwork, self).__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        # not sure of the whole padding thing
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        # sort this out, why hard coded to 60? [hadifawaz1999 answer: It was proposed like that in the original paper, check code in https://github.com/hfawaz/dl-4-tsc/blob/3ee62e16e118e4f5cfa86d01661846dfa75febfa/classifiers/cnn.py#L30]
        
        '''
        
        When the UCR archive was published in 2015, it contained 85 datasets, each dataset had
        different characteristics such as the time series length.

        The smallest dataset (refering to time series length) was the ItalyPowerDemand dataset
        with a length of 24,
        the second smallest one was SyntheticControl with a length of 60.
        
        The authors in [1], added the padding condition on the length 60 because it was the second smallest
        dataset (refering to time series length), SyntheticControl.

        Padding helps in the case of short time series because the samples would contain a small amount
        of information, which leads to a loss of information when applying convolutions without padding.
        This is due to the fact to negleting the side parts of the time series.

        Note that when the UCR archive had new datasets, 128 in 2018, more datasets of small length were added.
        For this reason, we keep the conditioning on 60 as length of input time series to add the padding in 
        the convolution.

        '''
        
        if input_shape[0] < 60:
            self.padding = ['same'] * self.n_layers

        x = input_layer

        for i in range(self.n_layers):

            conv = tf.keras.layers.Conv1D(filters=self.n_filters[i],
                                          kernel_size=self.kernel_sizes[i],
                                          strides=self.strides[i],
                                          padding=self.padding[i],
                                          dilation_rate=self.dilation_rate[i],
                                          activation=self.activation[i],
                                          use_bias=self.use_bias[i])(x)
            
            conv = tf.keras.layers.AveragePooling1D(pool_size=self.avg_pool_size[i])(conv)

            x = conv
    
        flatten_layer = tf.keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
