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
    dropout_rate: int, default = 0.1
        Randomly drops a certain percentage of neurons during training
    units: int , default = 500
        Number of neurons for a particular dense layer
    activation: String or Function, default = relu
        Specifies the activation function to use in the dense layers
    n_layers : int , default =3
        Specifies Number of hidden layers excluding input and output layer


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
        self, use_bias=True, dropout_rate=0.1, units=500, activation="relu", n_layers=3
    ):
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.units = units
        self.activation = activation
        self.n_layers = n_layers

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

        activation = getattr(keras.activations, self.activation, None)

        if activation is None:
            raise ValueError(f"Invalid activation function name:{self.activation}")

        x = input_layer_flattened

        for _ in range(self.n_layers):
            x = keras.layers.Dropout(self.dropout_rate)(x)
            x = keras.layers.Dense(
                self.units, activation=activation, use_bias=self.use_bias
            )(x)

        output_layer = keras.layers.Dropout(self.dropout_rate)(x)

        return input_layer, output_layer
