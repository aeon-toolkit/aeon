"""Informer Network for time series forecasting."""

__maintainer__ = [""]


from aeon.networks.base import BaseDeepLearningNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):

    from aeon.utils.networks.attention import (
        AttentionLayer,
        KerasProbAttention,
    )


class InformerNetwork(BaseDeepLearningNetwork):
    """
    TensorFlow implementation of the Informer network for time series forecasting.

    The Informer network is a Transformer-based architecture designed for
    long sequence time-series forecasting. It uses ProbSparse self-attention
    mechanism and distilling operation to reduce computational complexity.

    Parameters
    ----------
    encoder_input_len : int, default=96
        Encoder input sequence length.
    decoder_input_len : int, default=48
        Start token length for decoder.
    prediction_horizon : int, default=24
        Prediction sequence length.
    factor : int, default=5
        ProbSparse attention factor.
    model_dimension : int, default=512
        Model dimension.
    num_attention_heads : int, default=8
        Number of attention heads.
    encoder_layers : int, default=3
        Number of encoder layers.
    decoder_layers : int, default=2
        Number of decoder layers.
    feedforward_dim : int, default=512
        Feed forward network dimension.
    dropout : float, default=0.0
        Dropout rate.
    attention_type : str, default='prob'
        Attention mechanism type ('prob' or 'full').
    activation : str, default='gelu'
        Activation function.
    distil : bool, default=True
        Whether to use distilling operation.
    mix : bool, default=True
        Whether to use mix attention in decoder.

    References
    ----------
    .. [1] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W.
        (2021). Informer: Beyond efficient transformer for long sequence
        time-series forecasting. In Proceedings of the AAAI conference on
        artificial intelligence (Vol. 35, No. 12, pp. 11106-11115).
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "transformer",
    }

    def __init__(
        self,
        encoder_input_len: int = 96,
        decoder_input_len: int = 48,
        prediction_horizon: int = 24,
        factor: int = 5,
        model_dimension: int = 512,
        num_attention_heads: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        attention_type: str = "prob",
        activation: str = "gelu",
        distil: bool = True,
        mix: bool = True,
    ):
        self.encoder_input_len = encoder_input_len
        self.decoder_input_len = decoder_input_len
        self.prediction_horizon = prediction_horizon
        self.factor = factor
        self.model_dimension = model_dimension
        self.num_attention_heads = num_attention_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.attention_type = attention_type
        self.activation = activation
        self.distil = distil
        self.mix = mix

        super().__init__()

    def _token_embedding(self, input_tensor, c_in, model_dimension):
        """
        Token embedding layer using 1D convolution with causal padding.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor to be processed.
        c_in : int
            Number of input channels.
        model_dimension : int
            Dimension of the model (number of output filters).

        Returns
        -------
        tf.Tensor
            Output tensor after token embedding transformation.
        """
        import tensorflow as tf

        x = tf.keras.layers.Conv1D(
            filters=model_dimension,
            kernel_size=3,
            padding="causal",
            activation="linear",
        )(input_tensor)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def _positional_embedding(self, input_tensor, model_dimension, max_len=5000):
        """
        Positional embedding layer that computes positional encodings.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor to get positional embeddings for.
        model_dimension : int
            Dimension of the model.
        max_len : int, optional
            Maximum length of the sequence, by default 5000

        Returns
        -------
        tf.Tensor
            Positional encoding tensor matching input tensor's sequence length.
        """
        import math

        import numpy as np
        import tensorflow as tf

        # Compute the positional encodings
        pe = np.zeros((max_len, model_dimension), dtype=np.float32)
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(
            np.arange(0, model_dimension, 2, dtype=np.float32)
            * -(math.log(10000.0) / model_dimension)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Convert to tensor and add batch dimension
        pe_tensor = tf.expand_dims(tf.convert_to_tensor(pe), 0)

        # Return positional embeddings for the input tensor's sequence length
        return pe_tensor[:, : input_tensor.shape[1]]

    def _data_embedding(
        self,
        input_tensor,
        c_in,
        model_dimension,
        dropout=0.1,
        max_len=5000,
    ):
        """
        Combine token and positional embeddings for the input tensor.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor to be processed.
        c_in : int
            Number of input channels.
        model_dimension : int
            Dimension of the model (number of output filters).
        dropout : float, optional
            Dropout rate, by default 0.1
        max_len : int, optional
            Maximum length of the sequence for positional embedding

        Returns
        -------
        tf.Tensor
            Output tensor after data embedding transformation.
        """
        import tensorflow as tf

        # Get token embeddings
        token_emb = self._token_embedding(input_tensor, c_in, model_dimension)

        # Get positional embeddings
        pos_emb = self._positional_embedding(input_tensor, model_dimension, max_len)

        # Combine embeddings
        x = token_emb + pos_emb

        # Apply dropout
        x = tf.keras.layers.Dropout(dropout)(x)

        return x

    def _conv_layer(self, input_tensor, c_in):
        """
        Convolutional layer with batch normalization, ELU, and max pooling.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor to be processed.
        c_in : int
            Number of input channels (filters for the convolution).

        Returns
        -------
        tf.Tensor
            Output tensor after convolution and pooling operations.
        """
        import tensorflow as tf

        # Apply 1D convolution with causal padding
        x = tf.keras.layers.Conv1D(filters=c_in, kernel_size=3, padding="causal")(
            input_tensor
        )

        # Apply batch normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Apply ELU activation
        x = tf.keras.layers.ELU()(x)

        # Apply max pooling for downsampling
        x = tf.keras.layers.MaxPool1D(pool_size=3, strides=2)(x)

        return x

    def _attention_out(
        self,
        input_tensor,
        attention_type,
        mask_flag,
        model_dimension,
        num_attention_heads,
        factor=5,
        dropout=0.1,
        attn_mask=None,
    ):
        """
        Attention output layer applying either ProbAttention or FullAttention.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor for attention computation.
        attention_type : str
            Type of attention mechanism ('prob' or 'full').
        mask_flag : bool
            Whether to use attention masking.
        model_dimension : int
            Model dimension.
        num_attention_heads : int
            Number of attention heads.
        factor : int, optional
            Attention factor for ProbSparse attention, by default 5
        dropout : float, optional
            Dropout rate, by default 0.1
        attn_mask : tf.Tensor, optional
            Attention mask tensor, by default None

        Returns
        -------
        tf.Tensor
            Output tensor after attention computation.
        """
        import tensorflow as tf

        if attention_type == "prob":
            prob_attention = KerasProbAttention(
                mask_flag=mask_flag,
                factor=factor,
                attention_dropout=dropout,
            )

            output = AttentionLayer(
                attention=prob_attention,
                d_model=model_dimension,
                n_heads=num_attention_heads,
                d_keys=model_dimension // num_attention_heads,  # 512 // 8 = 64
                d_values=model_dimension // num_attention_heads,  # 512 // 8 = 64
            )(input_tensor, attn_mask=attn_mask)

        else:
            queries, keys, values = input_tensor
            output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_attention_heads,  # 8
                key_dim=model_dimension // num_attention_heads,  # 512 // 8 = 64
                value_dim=model_dimension // num_attention_heads,  # 512 // 8 = 64
                dropout=dropout,
                use_bias=True,
            )(
                query=queries,  # (32, 20, 512)
                key=keys,  # (32, 20, 512)
                value=values,  # (32, 20, 512)
                attention_mask=attn_mask,
                use_causal_mask=mask_flag,
            )

        return output

    def _encoder_layer(
        self,
        input_tensor,
        model_dimension,
        feedforward_dim=None,
        dropout=0.1,
        activation="relu",
        attn_mask=None,
        attention_type="prob",
        mask_flag=True,
        num_attention_heads=8,
        factor=5,
    ):
        """
        Apply encoder layer with multi-head attention and feed-forward network.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape [B, L, D] where B is batch size,
            L is sequence length, D is model dimension.
        model_dimension : int
            Model dimension (must match input tensor's last dimension).
        feedforward_dim : int, optional
            Feed-forward network dimension
        dropout : float, optional
            Dropout rate, by default 0.1
        activation : str, optional
            Activation function ('relu' or 'gelu'), by default "relu"
        attn_mask : tf.Tensor, optional
            Attention mask tensor, by default None

        Returns
        -------
        tf.Tensor
            Output tensor after encoder layer processing.
        """
        import tensorflow as tf

        # Set default feedforward_dim if not provided
        if feedforward_dim is None:
            feedforward_dim = 4 * model_dimension

        # Self-attention using the _attention_out function with parameters
        attn_output = self._attention_out(
            input_tensor=[input_tensor, input_tensor, input_tensor],
            attention_type=attention_type,
            mask_flag=mask_flag,
            model_dimension=model_dimension,
            num_attention_heads=num_attention_heads,
            factor=factor,
            dropout=dropout,
            attn_mask=attn_mask,
        )

        # Apply dropout and residual connection
        x = input_tensor + tf.keras.layers.Dropout(dropout)(attn_output)

        # First layer normalization
        x = tf.keras.layers.LayerNormalization()(x)

        # Store for second residual connection
        residual = x

        # Feed-forward network
        # First 1D convolution (expansion)
        y = tf.keras.layers.Conv1D(filters=feedforward_dim, kernel_size=1)(x)

        # Apply activation function
        if activation == "relu":
            y = tf.keras.layers.ReLU()(y)
        else:  # gelu
            y = tf.keras.layers.Activation("gelu")(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Second 1D convolution (compression back to d_model)
        y = tf.keras.layers.Conv1D(filters=model_dimension, kernel_size=1)(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Second residual connection and layer normalization
        output = tf.keras.layers.LayerNormalization()(residual + y)

        return output

    def _encoder(
        self,
        input_tensor,
        encoder_layers,
        model_dimension,
        feedforward_dim=None,
        dropout=0.1,
        activation="relu",
        attn_mask=None,
        attention_type="prob",
        mask_flag=True,
        num_attention_heads=8,
        factor=5,
        use_conv_layers=False,
        c_in=None,
        use_norm=True,
    ):
        """
        Apply encoder stack with multiple encoder layers and optional conv layers.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape [B, L, D]
        encoder_layers : int
            Number of encoder layers to stack.
        model_dimension : int
            Model dimension (must match input tensor's last dimension).
        feedforward_dim : int, optional
            Feed-forward network dimension
        dropout : float, optional
            Dropout rate, by default 0.1
        activation : str, optional
            Activation function ('relu' or 'gelu'), by default "relu"
        attn_mask : tf.Tensor, optional
            Attention mask tensor, by default None
        attention_type : str, optional
            Type of attention mechanism ('prob' or 'full')
        mask_flag : bool, optional
            Whether to use attention masking, by default True
        num_attention_heads : int, optional
            Number of attention heads, by default 8
        factor : int, optional
            Attention factor for ProbSparse attention, by default 5
        use_conv_layers : bool, optional
            Whether to use convolutional layers between encoder layers
        c_in : int, optional
            Number of input channels for convolutional layers
        use_norm : bool, optional
            Whether to apply final layer normalization, by default True

        Returns
        -------
        tf.Tensor
            Output tensor after encoder stack processing.
        """
        import tensorflow as tf

        # Set default values
        if c_in is None:
            c_in = model_dimension

        x = input_tensor

        # Apply encoder layers with optional convolutional layers
        if use_conv_layers:
            # Apply paired encoder and conv layers
            for _ in range(encoder_layers - 1):
                # Apply encoder layer
                x = self._encoder_layer(
                    input_tensor=x,
                    model_dimension=model_dimension,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    activation=activation,
                    attn_mask=attn_mask,
                    attention_type=attention_type,
                    mask_flag=mask_flag,
                    num_attention_heads=num_attention_heads,
                    factor=factor,
                )

                # Apply convolutional layer for downsampling
                x = self._conv_layer(
                    input_tensor=x,
                    c_in=c_in,
                )

            # Apply final encoder layer (without conv layer)
            x = self._encoder_layer(
                input_tensor=x,
                model_dimension=model_dimension,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                activation=activation,
                attn_mask=attn_mask,
                attention_type=attention_type,
                mask_flag=mask_flag,
                num_attention_heads=num_attention_heads,
                factor=factor,
            )

        else:
            # Apply only encoder layers without convolutional layers
            for _ in range(encoder_layers):
                x = self._encoder_layer(
                    input_tensor=x,
                    model_dimension=model_dimension,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    activation=activation,
                    attn_mask=attn_mask,
                    attention_type=attention_type,
                    mask_flag=mask_flag,
                    num_attention_heads=num_attention_heads,
                    factor=factor,
                )

        # Apply optional final layer normalization
        if use_norm:
            x = tf.keras.layers.LayerNormalization()(x)

        return x

    def _decoder_layer(
        self,
        input_tensor,
        cross_tensor,
        model_dimension,
        feedforward_dim=None,
        dropout=0.1,
        activation="relu",
        x_mask=None,
        cross_mask=None,
        self_attention_type="prob",
        cross_attention_type="prob",
        mask_flag=True,
        num_attention_heads=8,
        factor=5,
    ):
        """
        Apply decoder layer with self-attention, cross-attention, and FFN.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape [B, L, D]
        cross_tensor : tf.Tensor
            Cross-attention input tensor (encoder output) of shape [B, L_enc, D]
        model_dimension : int
            Model dimension (must match input tensor's last dimension).
        feedforward_dim : int, optional
            Feed-forward network dimension
        dropout : float, optional
            Dropout rate, by default 0.1
        activation : str, optional
            Activation function ('relu' or 'gelu'), by default "relu"
        x_mask : tf.Tensor, optional
            Self-attention mask tensor, by default None
        cross_mask : tf.Tensor, optional
            Cross-attention mask tensor, by default None
        self_attention_type : str, optional
            Type of self-attention mechanism ('prob' or 'full')
        cross_attention_type : str, optional
            Type of cross-attention mechanism ('prob' or 'full')
        mask_flag : bool, optional
            Whether to use attention masking, by default True
        num_attention_heads : int, optional
            Number of attention heads, by default 8
        factor : int, optional
            Attention factor for ProbSparse attention, by default 5

        Returns
        -------
        tf.Tensor
            Output tensor after decoder layer processing with same shape.
        """
        import tensorflow as tf

        # Set default feedforward_dim if not provided
        if feedforward_dim is None:
            feedforward_dim = 4 * model_dimension

        # Self-attention block
        self_attn_output = self._attention_out(
            input_tensor=[input_tensor, input_tensor, input_tensor],
            attention_type=self_attention_type,
            mask_flag=mask_flag,
            model_dimension=model_dimension,
            num_attention_heads=num_attention_heads,
            factor=factor,
            dropout=dropout,
            attn_mask=x_mask,
        )

        # Apply dropout and first residual connection
        x = input_tensor + tf.keras.layers.Dropout(dropout)(self_attn_output)

        # First layer normalization
        x = tf.keras.layers.LayerNormalization()(x)

        # Cross-attention block
        cross_attn_output = self._attention_out(
            input_tensor=[x, cross_tensor, cross_tensor],
            attention_type=cross_attention_type,
            mask_flag=mask_flag,
            model_dimension=model_dimension,
            num_attention_heads=num_attention_heads,
            factor=factor,
            dropout=dropout,
            attn_mask=cross_mask,
        )

        # Apply dropout and second residual connection
        x = x + tf.keras.layers.Dropout(dropout)(cross_attn_output)

        # Second layer normalization
        x = tf.keras.layers.LayerNormalization()(x)

        # Store for third residual connection
        residual = x

        # Feed-forward network
        # First 1D convolution (expansion)
        y = tf.keras.layers.Conv1D(filters=feedforward_dim, kernel_size=1)(x)

        # Apply activation function
        if activation == "relu":
            y = tf.keras.layers.ReLU()(y)
        else:  # gelu
            y = tf.keras.layers.Activation("gelu")(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Second 1D convolution (compression back to d_model)
        y = tf.keras.layers.Conv1D(filters=model_dimension, kernel_size=1)(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Third residual connection and final layer normalization
        output = tf.keras.layers.LayerNormalization()(residual + y)

        return output

    def _decoder(
        self,
        input_tensor,
        cross_tensor,
        decoder_layers,
        model_dimension,
        feedforward_dim=None,
        dropout=0.1,
        activation="relu",
        x_mask=None,
        cross_mask=None,
        self_attention_type="prob",
        cross_attention_type="prob",
        mask_flag=True,
        num_attention_heads=8,
        factor=5,
        use_norm=True,
    ):
        """
        Apply decoder stack with multiple decoder layers and optional normalization.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Decoder input tensor of shape [B, L_dec, D]
        cross_tensor : tf.Tensor
            Cross-attention input tensor (encoder output) of shape [B, L_enc, D]
        decoder_layers : int
            Number of decoder layers to stack.
        model_dimension : int
            Model dimension (must match input tensor's last dimension).
        feedforward_dim : int, optional
            Feed-forward network dimension
        dropout : float, optional
            Dropout rate, by default 0.1
        activation : str, optional
            Activation function ('relu' or 'gelu'), by default "relu"
        x_mask : tf.Tensor, optional
            Self-attention mask tensor for decoder, by default None
        cross_mask : tf.Tensor, optional
            Cross-attention mask tensor, by default None
        self_attention_type : str, optional
            Type of self-attention mechanism ('prob' or 'full')
        cross_attention_type : str, optional
            Type of cross-attention mechanism ('prob' or 'full')
        mask_flag : bool, optional
            Whether to use attention masking, by default True
        num_attention_heads : int, optional
            Number of attention heads, by default 8
        factor : int, optional
            Attention factor for ProbSparse attention, by default 5
        use_norm : bool, optional
            Whether to apply final layer normalization, by default True

        Returns
        -------
        tf.Tensor
            Output tensor after decoder stack processing.
        """
        import tensorflow as tf

        x = input_tensor

        # Apply multiple decoder layers
        for _ in range(decoder_layers):
            x = self._decoder_layer(
                input_tensor=x,
                cross_tensor=cross_tensor,
                model_dimension=model_dimension,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                activation=activation,
                x_mask=x_mask,
                cross_mask=cross_mask,
                self_attention_type=self_attention_type,
                cross_attention_type=cross_attention_type,
                mask_flag=mask_flag,
                num_attention_heads=num_attention_heads,
                factor=factor,
            )

        # Apply optional final layer normalization
        if use_norm:
            x = tf.keras.layers.LayerNormalization()(x)

        return x

    def _preprocess_time_series(
        self, data, encoder_input_len, decoder_input_len, prediction_horizon
    ):
        """
        Preprocess time series data of shape (None, n_timepoints, n_channels).

        Parameters
        ----------
        data : tf.Tensor
            Input tensor of shape (None, n_timepoints, n_channels)
        encoder_input_len : int
            Encoder input sequence length
        decoder_input_len : int
            Known decoder input length
        prediction_horizon : int
            Prediction length

        Returns
        -------
        tuple
            (x_enc, x_dec) where:
            - x_enc: Encoder input tensor of shape (None, encoder_input_len, n_channels)
            - x_dec: Decoder input tensor of shape (None,
            decoder_input_len + prediction_horizon, n_channels)
        """
        import tensorflow as tf

        # Get tensor dimensions - handle None batch dimension
        batch_size, n_timepoints, n_channels = data.shape

        # Encoder input: first seq_len timepoints
        x_enc = data[:, :encoder_input_len, :]  # (None, encoder_input_len, n_channels)

        # Decoder input construction
        x_dec_known = data[
            :, encoder_input_len - decoder_input_len : encoder_input_len, :
        ]  # (None, decoder_input_len, n_channels)

        # Unknown part: zeros for prediction horizon
        x_dec_pred = data[:, :prediction_horizon, :]

        # Concatenate known and prediction parts
        x_dec = tf.keras.layers.Concatenate(axis=1)([x_dec_known, x_dec_pred])

        return x_enc, x_dec

    def build_network(self, input_shape, **kwargs):
        """Build the complete Informer architecture for time series forecasting."""
        import tensorflow as tf

        # Get input dimensions
        n_timepoints, n_channels = input_shape

        input_data = tf.keras.layers.Input(
            shape=input_shape,
            name="time_series_input",
        )

        encoder_input, decoder_input = self._preprocess_time_series(
            data=input_data,
            encoder_input_len=self.encoder_input_len,
            decoder_input_len=self.decoder_input_len,
            prediction_horizon=self.prediction_horizon,
        )

        # Encoder embedding
        enc_embedded = self._data_embedding(
            input_tensor=encoder_input,
            c_in=n_channels,
            model_dimension=self.model_dimension,
            dropout=self.dropout,
            max_len=self.encoder_input_len,
        )

        # Encoder processing
        enc_output = self._encoder(
            input_tensor=enc_embedded,
            encoder_layers=self.encoder_layers,
            model_dimension=self.model_dimension,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            activation=self.activation,
            attention_type=self.attention_type,
            mask_flag=False,
            num_attention_heads=self.num_attention_heads,
            factor=self.factor,
            use_conv_layers=self.distil,
            c_in=self.model_dimension,
            use_norm=True,
        )

        # Decoder embedding
        dec_embedded = self._data_embedding(
            input_tensor=decoder_input,
            c_in=n_channels,
            model_dimension=self.model_dimension,
            dropout=self.dropout,
            max_len=self.decoder_input_len + self.prediction_horizon,
        )

        # Decoder processing
        dec_output = self._decoder(
            input_tensor=dec_embedded,
            cross_tensor=enc_output,
            decoder_layers=self.decoder_layers,
            model_dimension=self.model_dimension,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            activation=self.activation,
            self_attention_type=self.attention_type,
            cross_attention_type="full",
            mask_flag=self.mix,
            num_attention_heads=self.num_attention_heads,
            factor=self.factor,
            use_norm=True,
        )

        # Final projection to output dimension
        output = tf.keras.layers.Dense(n_channels, name="output_projection")(dec_output)

        # Extract only the prediction part (last out_len timesteps)
        output = output[:, -self.prediction_horizon :, :]

        return input_data, output
