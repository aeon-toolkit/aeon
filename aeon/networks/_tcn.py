"""Implementation of Temporal Convolutional Network (TCN).

Based on the paper "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling" by Bai et al. (2018).
"""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class TemporalConvolutionalNetwork(BaseDeepLearningNetwork):
    """Temporal Convolutional Network (TCN) for sequence modeling.

    A generic convolutional architecture for sequence modeling that combines:
    - Dilated convolutions for exponentially large receptive fields
    - Residual connections for training stability

    The TCN can take sequences of any length and map them to output sequences
    of the same length, making it suitable for autoregressive prediction tasks.

    Parameters
    ----------
    num_inputs : int
        Number of input channels/features in the input sequence.
    num_channels : list of int
        List specifying the number of output channels for each layer.
        The length determines the depth of the network.
    kernel_size : int, default=2
        Size of the convolutional kernel. Larger kernels can capture
        more local context but require more parameters.
    dropout : float, default=0.2
        Dropout rate applied after each convolutional layer for regularization.

    Notes
    -----
    The receptive field size grows exponentially with network depth due to
    dilated convolutions with dilation factors of 2^i for layer i.

    References
    ----------
    Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of
    generic convolutional and recurrent networks for sequence modeling.
    arXiv preprint arXiv:1803.01271.

    Examples
    --------
    >>> from aeon.networks._tcn import TemporalConvolutionalNetwork
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> import tensorflow as tf
    >>> X, y = make_example_3d_numpy(n_cases=8, n_channels=4, n_timepoints=150,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=42)
    >>> network = TemporalConvolutionalNetwork(num_inputs=4, num_channels=[8, 8])
    >>> input_layer, output = network.build_network(input_shape=(4, 150))
    >>> model = tf.keras.Model(inputs=input_layer, outputs=output)
    >>> model.compile(optimizer="adam", loss="mse")
    >>> model.fit(X, y, epochs=2, batch_size=2, verbose=0)  # doctest: +SKIP
    <keras.src.callbacks.History object ...>
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        num_inputs: int = 1,
        num_channels: list = [16] * 3,  # change to n_filters
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize the TCN architecture.

        Parameters
        ----------
        num_inputs : int
            Number of input channels/features.
        num_channels : list of int
            Number of output channels for each temporal block.
        kernel_size : int, default=2
            Size of convolutional kernels.
        dropout : float, default=0.2
            Dropout rate for regularization.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _conv1d_with_variable_padding(
        self,
        x,
        filters: int,
        kernel_size: int,
        padding_value: int,
        stride: int = 1,
        dilation_rate: int = 1,
    ):
        """Apply 1D convolution with variable padding for causal convolutions.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).
        filters : int
            Number of output filters.
        kernel_size : int
            Size of the convolutional kernel.
        padding_value : int
            Amount of padding to apply.
        stride : int, default=1
            Stride of the convolution.
        dilation_rate : int, default=1
            Dilation rate for dilated convolutions.

        Returns
        -------
        tf.Tensor
            Output tensor after convolution.
        """
        import tensorflow as tf

        # Transpose to Keras format (batch, sequence, channels)
        x_keras_format = tf.keras.layers.Permute((2, 1))(x)

        # Apply padding in sequence dimension
        padded_x = tf.keras.layers.ZeroPadding1D(padding=padding_value)(x_keras_format)

        # Create and apply convolution layer
        conv_layer = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            dilation_rate=dilation_rate,
            padding="valid",
        )

        # Apply convolution
        out = conv_layer(padded_x)

        # Transpose back to PyTorch format (batch, channels, sequence)
        return tf.keras.layers.Permute((2, 1))(out)

    def _chomp_1d(self, x, chomp_size: int):
        """Remove padding from the end of sequences to maintain causality.

        This operation ensures that the output at time t only depends on
        inputs from times 0 to t, preventing information leakage from future.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).
        chomp_size : int
            Number of time steps to remove from the end.

        Returns
        -------
        tf.Tensor
            Chomped tensor with reduced sequence length.
        """
        return x[:, :, :-chomp_size]

    def _temporal_block(
        self,
        x,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        training: bool = None,
    ):
        """Create a temporal block with dilated causal convolutions.

        Each temporal block consists of:
        1. Two dilated causal convolutions
        2. ReLU activations and dropout for regularization
        3. Residual connection with optional 1x1 convolution for dimension
           matching

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).
        n_inputs : int
            Number of input channels.
        n_outputs : int
            Number of output channels.
        kernel_size : int
            Size of convolutional kernels.
        stride : int
            Stride of convolutions (typically 1).
        dilation : int
            Dilation factor for dilated convolutions.
        padding : int
            Padding size to be chomped off.
        dropout : float, default=0.2
            Dropout rate for regularization.
        training : bool, optional
            Whether the model is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, n_outputs, sequence_length).
        """
        import tensorflow as tf

        # First convolution block
        out = self._conv1d_with_variable_padding(
            x, n_outputs, kernel_size, padding, stride, dilation
        )
        out = self._chomp_1d(out, padding)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Dropout(dropout)(out, training=training)

        # Second convolution block
        out = self._conv1d_with_variable_padding(
            out, n_outputs, kernel_size, padding, stride, dilation
        )
        out = self._chomp_1d(out, padding)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Dropout(dropout)(out, training=training)

        # Residual connection with optional dimension matching
        if n_inputs != n_outputs:
            res = self._conv1d_with_variable_padding(x, n_outputs, 1, 0, 1, 1)
        else:
            res = x

        # Add residual and apply final ReLU
        result = tf.keras.layers.Add()([out, res])
        return tf.keras.layers.ReLU()(result)

    def _temporal_conv_net(
        self,
        x,
        num_inputs: int,
        num_channels: list,
        kernel_size: int = 2,
        dropout: float = 0.2,
        training: bool = None,
    ):
        """Apply the complete Temporal Convolutional Network.

        Stacks multiple temporal blocks with exponentially increasing dilation
        factors to achieve a large receptive field efficiently.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).
        num_inputs : int
            Number of input channels.
        num_channels : list of int
            Number of output channels for each temporal block.
        kernel_size : int, default=2
            Size of convolutional kernels.
        dropout : float, default=0.2
            Dropout rate for regularization.
        training : bool, optional
            Whether the model is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor after applying all temporal blocks.
        """
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            x = self._temporal_block(
                x,
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout,
                training=training,
            )

        return x

    def build_network(self, input_shape: tuple, **kwargs) -> tuple:
        """Build the complete TCN architecture.

        Constructs a series of temporal blocks with exponentially increasing
        dilation factors to achieve a large receptive field efficiently.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data (sequence_length, num_features).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        tuple
            A tuple containing (input_layer, output_tensor) representing
            the complete network architecture.

        Notes
        -----
        The dilation factor for layer i is 2^i, which ensures exponential
        growth of the receptive field while maintaining computational
        efficiency.
        """
        import tensorflow as tf

        # Create input layer
        input_layer = tf.keras.layers.Input(shape=input_shape)

        # Transpose input to match the expected format (batch, channels, seq)
        x = input_layer

        # Apply TCN using the private function
        x = self._temporal_conv_net(
            x,
            num_inputs=self.num_inputs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        x = tf.keras.layers.Dense(input_shape[0])(x[:, -1, :])
        output = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1, keepdims=True), output_shape=(1,)
        )(x)
        return input_layer, output
