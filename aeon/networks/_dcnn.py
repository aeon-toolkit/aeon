import tensorflow as tf

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.networks import WeightNormalization


class DCNNEncoderNetwork(BaseDeepNetwork):
    def __init__(self, latent_dim, kernel_size, num_filters, model_depth, loss_function):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.loss_functions = loss_function
        self.model_depth = model_depth

    def build_network(self, input_shape):
        
        def DCNNLayer(inputs, num_filters, layer_num):
            _dilation = 2 ** layer_num
            _add = tf.keras.layers.Conv1D(num_filters, kernel_size=1)(inputs)
            x = WeightNormalization(
                tf.keras.layers.Conv1D(num_filters, dilation_rate=_dilation, padding="causal")
            )(inputs)
            x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
            x = WeightNormalization(
                tf.keras.layers.Conv1D(num_filters, dilation_rate=_dilation, padding="causal")
            )(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
            return tf.keras.layers.Add([x, _add])
        
        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(1, self.model_depth + 1):
            x = DCNNLayer(x, self.num_filters, i)
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        output_layer = tf.keras.layers.Dense(self.latent_dim)(x)

        return tf.keras.Model(inputs=input_layer, outputs = output_layer)
    