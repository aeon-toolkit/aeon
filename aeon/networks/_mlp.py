"""Multi Layer Perceptron Network (MLPNetwork)."""

__maintainer__ = ["hadifawaz1999"]


from aeon.networks.base import BaseDeepLearningNetwork


class MLPNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for a MLP.

    Adapted from the implementation used in [1]_

    Parameters
    ----------
    use_bias : bool, default = True
        Condition on whether or not to use bias values for dense layers.

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    References
    ----------
    .. [1]  Wang et al. Time series classification from scratch with deep neural
    networks: A strong baseline, IJCNN, 2017.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        use_bias=True,
    ):
        self.use_bias = use_bias

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras

        # flattened because multivariate should be on same axis
        input_layer = keras.layers.Input(input_shape)
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation="relu", use_bias=self.use_bias)(
            layer_1
        )

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation="relu", use_bias=self.use_bias)(
            layer_2
        )

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation="relu", use_bias=self.use_bias)(
            layer_3
        )

        output_layer = keras.layers.Dropout(0.3)(layer_3)

        return input_layer, output_layer
