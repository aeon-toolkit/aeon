"""Implementation of Temporal Convolutional Network (TCN)."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class TCNNetwork(BaseDeepLearningNetwork):
    """Temporal Convolutional Network (TCN) for sequence modeling.

    A generic convolutional architecture for sequence modeling that combines:
    - Dilated convolutions for exponentially large receptive fields
    - Residual connections for training stability

    The TCN can take sequences of any length and map them to output sequences
    of the same length, making it suitable for autoregressive prediction tasks.

    Parameters
    ----------
    n_blocks : list of int
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
    .. [1]  Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of
    generic convolutional and recurrent networks for sequence modeling.
    arXiv preprint arXiv:1803.01271.

    Examples
    --------
    >>> from aeon.networks._tcn import TCNNetwork
    >>> network = TCNNetwork(n_blocks=[8, 8])
    >>> input_layer, output = network.build_network(input_shape=(150, 4))
    >>> input_layer.shape, output.shape
    ((None, 150, 4), (None, 4))
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        n_blocks: list = [16] * 3,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize the TCN architecture.

        Parameters
        ----------
        n_blocks : list of int
            Number of output channels for each temporal block.
        kernel_size : int, default=2
            Size of convolutional kernels.
        dropout : float, default=0.2
            Dropout rate for regularization.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _conv1d_with_variable_padding(
        self,
        input_tensor,
        n_filters: int,
        kernel_size: int,
        padding_value: int,
        strides: int = 1,
        dilation_rate: int = 1,
    ):
        """Apply 1D convolution with variable padding for causal convolutions.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape (batch_size, n_timepoints, n_channels).
        n_filters : int
            Number of output filters.
        kernel_size : int
            Size of the convolutional kernel.
        padding_value : int
            Amount of padding to apply.
        strides : int, default=1
            Stride of the convolution.
        dilation_rate : int, default=1
            Dilation rate for dilated convolutions.

        Returns
        -------
        tf.Tensor
            Output tensor after convolution.
        """
        import tensorflow as tf

        # Apply padding in sequence dimension
        padded_x = tf.keras.layers.ZeroPadding1D(padding=padding_value)(input_tensor)

        # Create and apply convolution layer
        conv_layer = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding="valid",
        )

        # Apply convolution
        out = conv_layer(padded_x)

        return out

    def _chomp(self, input_tensor, chomp_size: int):
        """Remove padding from the end of sequences to maintain causality.

        This operation ensures that the output at time t only depends on
        inputs from times 0 to t, preventing information leakage from future.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape (batch_size, sequence_length, channels).
        chomp_size : int
            Number of time steps to remove from the end.

        Returns
        -------
        tf.Tensor
            Chomped tensor with reduced sequence length.
        """
        return input_tensor[:, :-chomp_size, :]

    def _temporal_block(
        self,
        input_tensor,
        n_inputs: int,
        n_filters: int,
        kernel_size: int,
        strides: int,
        dilation_rate: int,
        padding_value: int,
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
        input_tensor : tf.Tensor
            Input tensor of shape (batch_size, sequence_length, channels).
        n_inputs : int
            Number of input channels.
        n_filters : int
            Number of output filters.
        kernel_size : int
            Size of convolutional kernels.
        strides : int
            Stride of convolutions (typically 1).
        dilation_rate : int
            Dilation factor for dilated convolutions.
        padding_value : int
            Padding size to be chomped off.
        dropout : float, default=0.2
            Dropout rate for regularization.
        training : bool, optional
            Whether the model is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, sequence_length, n_filters).
        """
        import tensorflow as tf

        # First convolution block
        out = self._conv1d_with_variable_padding(
            input_tensor, n_filters, kernel_size, padding_value, strides, dilation_rate
        )
        out = self._chomp(out, padding_value)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Dropout(dropout)(out, training=training)

        # Second convolution block
        out = self._conv1d_with_variable_padding(
            out, n_filters, kernel_size, padding_value, strides, dilation_rate
        )
        out = self._chomp(out, padding_value)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Dropout(dropout)(out, training=training)

        # Residual connection with optional dimension matching
        if n_inputs != n_filters:
            res = self._conv1d_with_variable_padding(
                input_tensor=input_tensor,
                n_filters=n_filters,
                kernel_size=1,
                padding_value=0,
                strides=1,
                dilation_rate=1,
            )
        else:
            res = input_tensor

        # Add residual and apply final ReLU
        result = tf.keras.layers.Add()([out, res])
        return tf.keras.layers.ReLU()(result)

    def _temporal_conv_net(
        self,
        input_tensor,
        n_inputs: int,
        n_blocks: list,
        kernel_size: int = 2,
        dropout: float = 0.2,
        training: bool = None,
    ):
        """Apply the complete Temporal Convolutional Network.

        Stacks multiple temporal blocks with exponentially increasing dilation
        factors to achieve a large receptive field efficiently.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).
        n_inputs : int
            Number of input channels.
        n_blocks : list of int
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
        num_levels = len(n_blocks)
        for i in range(num_levels):
            dilation_rate = 2**i
            in_channels = n_inputs if i == 0 else n_blocks[i - 1]
            out_channels = n_blocks[i]
            padding_value = (kernel_size - 1) * dilation_rate

            input_tensor = self._temporal_block(
                input_tensor,
                n_inputs=in_channels,
                n_filters=out_channels,
                kernel_size=kernel_size,
                strides=1,
                dilation_rate=dilation_rate,
                padding_value=padding_value,
                dropout=dropout,
                training=training,
            )

        return input_tensor

    def build_network(self, input_shape: tuple, **kwargs) -> tuple:
        """Build the complete TCN architecture.

        Constructs a series of temporal blocks with exponentially increasing
        dilation factors to achieve a large receptive field efficiently.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data (n_timepoints, n_channels).
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

        # Transpose input to match the expected format (batch, n_timepoints, n_channels)
        x = input_layer
        n_inputs = input_shape[1]  # input_shape is of shape (n_timepoints, n_channels)

        # Apply TCN using the private function
        x = self._temporal_conv_net(
            x,
            n_inputs=n_inputs,
            n_blocks=self.n_blocks,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        output = tf.keras.layers.Dense(input_shape[1])(x[:, -1, :])
        # output = tf.keras.layers.Dense(1)(x)
        return input_layer, output
