# -*- coding: utf-8 -*-

__author__ = ["Ali Ismail-Fawaz"]

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class EncoderNetwork(BaseDeepNetwork):

    '''
    
    Establish the network structure for an Encoder.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_sizes    : array of int, default = [5, 11, 21]
        specifying the length of the 1D convolution windows
    n_filters       : array of int, default = [128, 256, 512]
        specifying the number of 1D convolution filters used for each layer,
        the shape of this array should be the same as kernel_sizes
    max_pool_size   : int, default = 2
        size of the max pooling windows
    activation      : string, default = sigmoid
        keras activation function
    dropout_proba   : float, default = 0.2
        specifying the dropout layer probability
    padding         : string, default = same
        specifying the type of padding used for the 1D convolution
    strides         : int, default = 1
        specifying the sliding rate of the 1D convolution filter
    fc_units        : int, default = 256
        specifying the number of units in the hiddent fully connected layer used in the EncoderNetwork
    random_state    : int, default = 0
        seed to any needed random actions

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    .. [1] Serrà et al. Towards a Universal Neural Network Encoder for Time Series 
    In proceedings International Conference of the Catalan Association for Artificial Intelligence, 120--129 2018.
    
    '''

    _tags = {"python_dependencies": ["tensorflow", "tensorflow_addons "]}

    def __init__(
        self,
        kernel_sizes=[5, 11, 21],
        n_filters=[128, 256, 512],
        dropout_proba=0.2,
        max_pool_size=2,
        activation="sigmoid",
        padding="same",
        strides=1,
        fc_units=256,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        self.random_state = random_state

        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.padding = padding
        self.strides = strides

        self.max_pool_size = max_pool_size
        self.activation = activation
        self.dropout_proba = dropout_proba

        self.fc_units = fc_units

        super(EncoderNetwork, self).__init__()

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

        import tensorflow as tf
        import tensorflow_addons as tfa

        # iunput layer

        input_layer = tf.keras.layers.Input(input_shape)
        
        x = input_layer

        for i, kernel_size in enumerate(self.kernel_sizes):

            conv = tf.keras.layers.Conv1D(filters=self.n_filters[i],
                                          kernel_size=kernel_size,
                                          padding=self.padding,
                                          strides=self.strides)(x)
            
            conv = tfa.layers.InstanceNormalization()(conv)
            conv = tf.keras.layers.PReLU(shared_axes=[1])(conv)
            conv = tf.keras.layers.Dropout(self.dropout_proba)(conv)

            if i < len(self.kernel_sizes) - 1:
                conv = tf.keras.layers.MaxPool1D(pool_size=self.max_pool_size)(conv)
            
            x = conv
        
        # split attention

        split_index = self.n_filters[-1] // 2

        attention_multiplier_1 = tf.keras.layers.Softmax(tf.keras.layers.Lambda(lambda x: x[:,:,:split_index])(conv))
        attention_multiplier_2 = tf.keras.layers.Lambda(lambda x: x[:,:,split_index:])(conv)

        # attention mechanism

        attention = tf.keras.layers.Multiply()([attention_multiplier_1, attention_multiplier_2])

        # add fully connected hidden layer

        hidden_fc_layer = tf.keras.layers.Dense(units=self.fc_units, activation=self.activation)(attention)
        hidden_fc_layer = tfa.layers.InstanceNormalization()(hidden_fc_layer)

        # output layer before classification layer

        flatten_layer = tf.keras.layers.Flatten()(hidden_fc_layer)
        
        return input_layer, flatten_layer