"""Informer Network for time series forecasting."""

__maintainer__ = [""]

from typing import Optional

from aeon.networks.base import BaseDeepLearningNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

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
    seq_len : int, default=96
        Input sequence length.
    label_len : int, default=48
        Start token length for decoder.
    out_len : int, default=24
        Prediction sequence length.
    factor : int, default=5
        ProbSparse attention factor.
    d_model : int, default=512
        Model dimension.
    n_heads : int, default=8
        Number of attention heads.
    e_layers : int, default=3
        Number of encoder layers.
    d_layers : int, default=2
        Number of decoder layers.
    d_ff : int, default=512
        Feed forward network dimension.
    dropout : float, default=0.0
        Dropout rate.
    attn : str, default='prob'
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
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        seq_len: int = 96,
        label_len: int = 48,
        out_len: int = 24,
        factor: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
        attn: str = "prob",
        activation: str = "gelu",
        distil: bool = True,
        mix: bool = True,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.attn = attn
        self.activation = activation
        self.distil = distil
        self.mix = mix

        super().__init__()

    def _token_embedding(
        self, input_tensor: tf.Tensor, c_in: int, d_model: int
    ) -> tf.Tensor:
        """
        Token embedding layer using 1D convolution with causal padding.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor to be processed.
        c_in : int
            Number of input channels.
        d_model : int
            Dimension of the model (number of output filters).

        Returns
        -------
        tf.Tensor
            Output tensor after token embedding transformation.
        """
        import tensorflow as tf

        x = tf.keras.layers.Conv1D(
            filters=d_model, kernel_size=3, padding="causal", activation="linear"
        )(input_tensor)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def _positional_embedding(
        self, input_tensor: tf.Tensor, d_model: int, max_len: int = 5000
    ) -> tf.Tensor:
        """
        Positional embedding layer that computes positional encodings.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor to get positional embeddings for.
        d_model : int
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
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Convert to tensor and add batch dimension
        pe_tensor = tf.expand_dims(tf.convert_to_tensor(pe), 0)

        # Return positional embeddings for the input tensor's sequence length
        return pe_tensor[:, : input_tensor.shape[1]]

    def _data_embedding(
        self,
        input_tensor: tf.Tensor,
        c_in: int,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> tf.Tensor:
        """
        Combine token and positional embeddings for the input tensor.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor to be processed.
        c_in : int
            Number of input channels.
        d_model : int
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
        token_emb = self._token_embedding(input_tensor, c_in, d_model)

        # Get positional embeddings
        pos_emb = self._positional_embedding(input_tensor, d_model, max_len)

        # Combine embeddings
        x = token_emb + pos_emb

        # Apply dropout
        x = tf.keras.layers.Dropout(dropout)(x)

        return x

    def _conv_layer(self, input_tensor: tf.Tensor, c_in: int) -> tf.Tensor:
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
        input_tensor: tf.Tensor,
        attention_type: str,
        mask_flag: bool,
        d_model: int,
        n_heads: int,
        factor: int = 5,
        dropout: float = 0.1,
        attn_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
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
        d_model : int
            Model dimension.
        n_heads : int
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
                d_model=d_model,
                n_heads=n_heads,
                d_keys=d_model // n_heads,  # 512 // 8 = 64
                d_values=d_model // n_heads,  # 512 // 8 = 64
            )(input_tensor, attn_mask=attn_mask)

        else:
            queries, keys, values = input_tensor
            output = tf.keras.layers.MultiHeadAttention(
                num_heads=n_heads,  # 8
                key_dim=d_model // n_heads,  # 512 // 8 = 64
                value_dim=d_model // n_heads,  # 512 // 8 = 64
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
        input_tensor: tf.Tensor,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        attn_mask: Optional[tf.Tensor] = None,
        attention_type: str = "prob",
        mask_flag: bool = True,
        n_heads: int = 8,
        factor: int = 5,
    ) -> tf.Tensor:
        """
        Apply encoder layer with multi-head attention and feed-forward network.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape [B, L, D] where B is batch size,
            L is sequence length, D is model dimension.
        d_model : int
            Model dimension (must match input tensor's last dimension).
        d_ff : int, optional
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

        # Set default d_ff if not provided
        if d_ff is None:
            d_ff = 4 * d_model

        # Self-attention using the _attention_out function with parameters
        attn_output = self._attention_out(
            input_tensor=[input_tensor, input_tensor, input_tensor],
            attention_type=attention_type,
            mask_flag=mask_flag,
            d_model=d_model,
            n_heads=n_heads,
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
        y = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1)(x)

        # Apply activation function
        if activation == "relu":
            y = tf.keras.layers.ReLU()(y)
        else:  # gelu
            y = tf.keras.layers.Activation("gelu")(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Second 1D convolution (compression back to d_model)
        y = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Second residual connection and layer normalization
        output = tf.keras.layers.LayerNormalization()(residual + y)

        return output

    def _encoder(
        self,
        input_tensor: tf.Tensor,
        e_layers: int,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        attn_mask: Optional[tf.Tensor] = None,
        attention_type: str = "prob",
        mask_flag: bool = True,
        n_heads: int = 8,
        factor: int = 5,
        use_conv_layers: bool = False,
        c_in: Optional[int] = None,
        use_norm: bool = True,
    ) -> tf.Tensor:
        """
        Apply encoder stack with multiple encoder layers and optional conv layers.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape [B, L, D]
        e_layers : int
            Number of encoder layers to stack.
        d_model : int
            Model dimension (must match input tensor's last dimension).
        d_ff : int, optional
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
        n_heads : int, optional
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
            c_in = d_model

        x = input_tensor

        # Apply encoder layers with optional convolutional layers
        if use_conv_layers:
            # Apply paired encoder and conv layers
            for _ in range(e_layers - 1):
                # Apply encoder layer
                x = self._encoder_layer(
                    input_tensor=x,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    attn_mask=attn_mask,
                    attention_type=attention_type,
                    mask_flag=mask_flag,
                    n_heads=n_heads,
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
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                attn_mask=attn_mask,
                attention_type=attention_type,
                mask_flag=mask_flag,
                n_heads=n_heads,
                factor=factor,
            )

        else:
            # Apply only encoder layers without convolutional layers
            for _ in range(e_layers):
                x = self._encoder_layer(
                    input_tensor=x,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    attn_mask=attn_mask,
                    attention_type=attention_type,
                    mask_flag=mask_flag,
                    n_heads=n_heads,
                    factor=factor,
                )

        # Apply optional final layer normalization
        if use_norm:
            x = tf.keras.layers.LayerNormalization()(x)

        return x

    def _decoder_layer(
        self,
        input_tensor: tf.Tensor,
        cross_tensor: tf.Tensor,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        x_mask: Optional[tf.Tensor] = None,
        cross_mask: Optional[tf.Tensor] = None,
        self_attention_type: str = "prob",
        cross_attention_type: str = "prob",
        mask_flag: bool = True,
        n_heads: int = 8,
        factor: int = 5,
    ) -> tf.Tensor:
        """
        Apply decoder layer with self-attention, cross-attention, and FFN.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape [B, L, D]
        cross_tensor : tf.Tensor
            Cross-attention input tensor (encoder output) of shape [B, L_enc, D]
        d_model : int
            Model dimension (must match input tensor's last dimension).
        d_ff : int, optional
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
        n_heads : int, optional
            Number of attention heads, by default 8
        factor : int, optional
            Attention factor for ProbSparse attention, by default 5

        Returns
        -------
        tf.Tensor
            Output tensor after decoder layer processing with same shape.
        """
        import tensorflow as tf

        # Set default d_ff if not provided
        if d_ff is None:
            d_ff = 4 * d_model

        # Self-attention block
        self_attn_output = self._attention_out(
            input_tensor=[input_tensor, input_tensor, input_tensor],
            attention_type=self_attention_type,
            mask_flag=mask_flag,
            d_model=d_model,
            n_heads=n_heads,
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
            d_model=d_model,
            n_heads=n_heads,
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
        y = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1)(x)

        # Apply activation function
        if activation == "relu":
            y = tf.keras.layers.ReLU()(y)
        else:  # gelu
            y = tf.keras.layers.Activation("gelu")(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Second 1D convolution (compression back to d_model)
        y = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)(y)

        # Apply dropout
        y = tf.keras.layers.Dropout(dropout)(y)

        # Third residual connection and final layer normalization
        output = tf.keras.layers.LayerNormalization()(residual + y)

        return output

    def _decoder(
        self,
        input_tensor: tf.Tensor,
        cross_tensor: tf.Tensor,
        d_layers: int,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        x_mask: Optional[tf.Tensor] = None,
        cross_mask: Optional[tf.Tensor] = None,
        self_attention_type: str = "prob",
        cross_attention_type: str = "prob",
        mask_flag: bool = True,
        n_heads: int = 8,
        factor: int = 5,
        use_norm: bool = True,
    ) -> tf.Tensor:
        """
        Apply decoder stack with multiple decoder layers and optional normalization.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Decoder input tensor of shape [B, L_dec, D]
        cross_tensor : tf.Tensor
            Cross-attention input tensor (encoder output) of shape [B, L_enc, D]
        d_layers : int
            Number of decoder layers to stack.
        d_model : int
            Model dimension (must match input tensor's last dimension).
        d_ff : int, optional
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
        n_heads : int, optional
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
        for _ in range(d_layers):
            x = self._decoder_layer(
                input_tensor=x,
                cross_tensor=cross_tensor,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                x_mask=x_mask,
                cross_mask=cross_mask,
                self_attention_type=self_attention_type,
                cross_attention_type=cross_attention_type,
                mask_flag=mask_flag,
                n_heads=n_heads,
                factor=factor,
            )

        # Apply optional final layer normalization
        if use_norm:
            x = tf.keras.layers.LayerNormalization()(x)

        return x

    def build_network(
        self, input_shape: tuple[int, int], **kwargs
    ) -> tuple[list[tf.Tensor], tf.Tensor]:
        """Build the complete Informer architecture for time series forecasting."""
        import tensorflow as tf

        # Get input dimensions
        n_timepoints, n_channels = input_shape

        # hardcode batch_size for now
        batch_size = 32

        # Create input layers for encoder and decoder
        encoder_input = tf.keras.layers.Input(
            shape=(self.seq_len, n_channels),
            name="encoder_input",
            batch_size=batch_size,
        )

        decoder_input = tf.keras.layers.Input(
            shape=(self.label_len + self.out_len, n_channels),
            name="decoder_input",
            batch_size=batch_size,
        )

        # Encoder embedding
        enc_embedded = self._data_embedding(
            input_tensor=encoder_input,
            c_in=n_channels,
            d_model=self.d_model,
            dropout=self.dropout,
            max_len=self.seq_len,
        )

        # Encoder processing
        enc_output = self._encoder(
            input_tensor=enc_embedded,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            attention_type=self.attn,
            mask_flag=False,
            n_heads=self.n_heads,
            factor=self.factor,
            use_conv_layers=self.distil,
            c_in=self.d_model,
            use_norm=True,
        )

        # Decoder embedding
        dec_embedded = self._data_embedding(
            input_tensor=decoder_input,
            c_in=n_channels,
            d_model=self.d_model,
            dropout=self.dropout,
            max_len=self.label_len + self.out_len,
        )

        # Decoder processing
        dec_output = self._decoder(
            input_tensor=dec_embedded,
            cross_tensor=enc_output,
            d_layers=self.d_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            self_attention_type=self.attn,
            cross_attention_type="full",
            mask_flag=self.mix,
            n_heads=self.n_heads,
            factor=self.factor,
            use_norm=True,
        )

        # Final projection to output dimension
        output = tf.keras.layers.Dense(n_channels, name="output_projection")(dec_output)

        # Extract only the prediction part (last out_len timesteps)
        output = output[:, -self.out_len :, :]

        # Create the model with both encoder and decoder inputs
        inputs = [encoder_input, decoder_input]

        return inputs, output
