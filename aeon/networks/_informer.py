"""Informer Network for time series forecasting."""

__maintainer__ = [""]

from aeon.networks.base import BaseDeepLearningNetwork


class InformerNetwork(BaseDeepLearningNetwork):
    """
    TensorFlow implementation of the Informer network for time series forecasting.

    The Informer network is a Transformer-based architecture designed for
    long sequence time-series forecasting. It uses ProbSparse self-attention
    mechanism and distilling operation to reduce computational complexity.

    Parameters
    ----------
    enc_in : int, default=7
        Number of encoder input features.
    dec_in : int, default=7
        Number of decoder input features.
    c_out : int, default=7
        Number of output features.
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
    embed : str, default='fixed'
        Embedding type.
    freq : str, default='h'
        Time frequency encoding.
    activation : str, default='gelu'
        Activation function.
    output_attention : bool, default=False
        Whether to output attention weights.
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
        "structure": "encoder-decoder",
    }

    def __init__(
        self,
        enc_in=7,
        dec_in=7,
        c_out=7,
        seq_len=96,
        label_len=48,
        out_len=24,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
    ):
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
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
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention
        self.distil = distil
        self.mix = mix

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """
        Construct the Informer network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        input_layer : keras.layers.Input
            The input layer of the network.
        output_layer : keras.layers.Layer
            The output layer of the network.
        """
        import tensorflow as tf

        # Input layers
        x_enc = tf.keras.layers.Input(
            shape=(self.seq_len, self.enc_in), name="encoder_input"
        )
        x_mark_enc = tf.keras.layers.Input(shape=(self.seq_len, 4), name="encoder_mark")
        x_dec = tf.keras.layers.Input(
            shape=(self.label_len + self.out_len, self.dec_in), name="decoder_input"
        )
        x_mark_dec = tf.keras.layers.Input(
            shape=(self.label_len + self.out_len, 4), name="decoder_mark"
        )

        # Encoder embedding
        enc_embedding = self._data_embedding(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        enc_out = enc_embedding([x_enc, x_mark_enc])

        # Encoder
        encoder = self._build_encoder()
        enc_out, attns = encoder(enc_out)

        # Decoder embedding
        dec_embedding = self._data_embedding(
            self.dec_in, self.d_model, self.embed, self.freq, self.dropout
        )
        dec_out = dec_embedding([x_dec, x_mark_dec])

        # Decoder
        decoder = self._build_decoder()
        dec_out = decoder([dec_out, enc_out])

        # Final projection
        projection = tf.keras.layers.Dense(self.c_out, use_bias=True, name="projection")
        dec_out = projection(dec_out)

        # Extract prediction sequence
        output = tf.keras.layers.Lambda(
            lambda x: x[:, -self.out_len :, :], name="prediction_slice"
        )(dec_out)

        # Create model inputs list
        inputs = [x_enc, x_mark_enc, x_dec, x_mark_dec]

        if self.output_attention:
            outputs = [output, attns]
        else:
            outputs = output

        return inputs, outputs

    def _positional_embedding(self, d_model, max_len=5000):
        """Create positional embedding layer."""
        import math

        import numpy as np
        import tensorflow as tf

        # Compute the positional encodings once in log space
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :]  # Add batch dimension

        # Create constant tensor
        pe_tensor = tf.constant(pe, dtype=tf.float32)

        def positional_function(x):
            seq_len = tf.shape(x)[1]
            return pe_tensor[:, :seq_len, :]

        return positional_function

    def _token_embedding(self, c_in, d_model):
        """Create token embedding layer."""
        import tensorflow as tf

        token_conv = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=3,
            padding="same",
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )

        def token_function(x):
            return token_conv(x)

        return token_function

    def _fixed_embedding(self, c_in, d_model):
        """Create fixed embedding layer."""
        import math

        import numpy as np
        import tensorflow as tf

        # Create fixed sinusoidal embeddings
        w = np.zeros((c_in, d_model), dtype=np.float32)
        position = np.arange(0, c_in, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )

        w[:, 0::2] = np.sin(position * div_term)
        w[:, 1::2] = np.cos(position * div_term)

        # Create embedding layer with fixed weights
        embedding = tf.keras.layers.Embedding(
            input_dim=c_in,
            output_dim=d_model,
            embeddings_initializer="zeros",
            trainable=False,
        )

        def fixed_function(x):
            # Initialize weights if not already done
            if not embedding.built:
                embedding.build((None,))
                embedding.embeddings.assign(w)
            return tf.stop_gradient(embedding(x))

        return fixed_function

    def _temporal_embedding(self, d_model, embed_type, freq):
        """Create temporal embedding layer."""
        import tensorflow as tf

        # Define embedding sizes
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # Choose embedding type
        if embed_type == "fixed":
            minute_embed = (
                self._fixed_embedding(minute_size, d_model) if freq == "t" else None
            )
            hour_embed = self._fixed_embedding(hour_size, d_model)
            weekday_embed = self._fixed_embedding(weekday_size, d_model)
            day_embed = self._fixed_embedding(day_size, d_model)
            month_embed = self._fixed_embedding(month_size, d_model)
        else:
            minute_embed = (
                tf.keras.layers.Embedding(minute_size, d_model) if freq == "t" else None
            )
            hour_embed = tf.keras.layers.Embedding(hour_size, d_model)
            weekday_embed = tf.keras.layers.Embedding(weekday_size, d_model)
            day_embed = tf.keras.layers.Embedding(day_size, d_model)
            month_embed = tf.keras.layers.Embedding(month_size, d_model)

        def temporal_function(x):
            x = tf.cast(x, tf.int32)

            minute_x = minute_embed(x[:, :, 4]) if minute_embed is not None else 0.0
            hour_x = hour_embed(x[:, :, 3])
            weekday_x = weekday_embed(x[:, :, 2])
            day_x = day_embed(x[:, :, 1])
            month_x = month_embed(x[:, :, 0])

            return hour_x + weekday_x + day_x + month_x + minute_x

        return temporal_function

    def _time_feature_embedding(self, d_model, embed_type, freq):
        """Create time feature embedding layer."""
        import tensorflow as tf

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]

        embed_layer = tf.keras.layers.Dense(d_model)

        def time_feature_function(x):
            return embed_layer(x)

        return time_feature_function

    def _data_embedding(self, c_in, d_model, embed_type, freq, dropout):
        """Create data embedding layer."""
        import tensorflow as tf

        # Create embedding components
        value_embedding = self._token_embedding(c_in, d_model)
        position_embedding = self._positional_embedding(d_model)

        if embed_type != "timeF":
            temporal_embedding = self._temporal_embedding(d_model, embed_type, freq)
        else:
            temporal_embedding = self._time_feature_embedding(d_model, embed_type, freq)

        dropout_layer = tf.keras.layers.Dropout(dropout)

        def embedding_function(inputs, training=None):
            x, x_mark = inputs

            value_emb = value_embedding(x)
            pos_emb = position_embedding(x)
            temporal_emb = temporal_embedding(x_mark)

            embeddings = value_emb + pos_emb + temporal_emb
            return dropout_layer(embeddings, training=training)

        return embedding_function

    def _build_encoder(self):
        """Build the encoder stack with attention layers."""
        import tensorflow as tf

        # Choose attention type
        if self.attn == "prob":
            Attn = self._prob_attention(
                False, self.factor, self.dropout, self.output_attention
            )
        else:
            Attn = self._full_attention(
                False, self.factor, self.dropout, self.output_attention
            )

        # Build encoder layers
        encoder_layers = []
        for l in range(self.e_layers):
            attention_layer = self._attention_layer(
                Attn, self.d_model, self.n_heads, mix=False
            )
            encoder_layer = self._encoder_layer(
                attention_layer, self.d_model, self.d_ff, self.dropout, self.activation
            )
            encoder_layers.append(encoder_layer)

        # Build conv layers for distilling
        conv_layers = None
        if self.distil:
            conv_layers = []
            for l in range(self.e_layers - 1):
                conv_layer = self._conv_layer(self.d_model)
                conv_layers.append(conv_layer)

        # Normalization layer
        norm_layer = tf.keras.layers.LayerNormalization()

        def encoder_function(x, attn_mask=None, training=None):
            # x [B, L, D]
            attns = []

            if conv_layers is not None:
                # Process with both attention and conv layers
                for attn_layer, conv_layer in zip(encoder_layers, conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask, training=training)
                    x = conv_layer(x, training=training)
                    attns.append(attn)

                # Final attention layer
                x, attn = encoder_layers[-1](x, attn_mask=attn_mask, training=training)
                attns.append(attn)
            else:
                # Process with only attention layers
                for attn_layer in encoder_layers:
                    x, attn = attn_layer(x, attn_mask=attn_mask, training=training)
                    attns.append(attn)

            if norm_layer is not None:
                x = norm_layer(x, training=training)

            return x, attns

        return encoder_function

    def _build_decoder(self):
        """Build the decoder stack with attention layers."""
        import tensorflow as tf

        # Build decoder layers
        decoder_layers = []
        for l in range(self.d_layers):
            # Self-attention (with mask)
            self_attn = (
                self._prob_attention(True, self.factor, self.dropout, False)
                if self.attn == "prob"
                else self._full_attention(True, self.factor, self.dropout, False)
            )
            self_attention_layer = self._attention_layer(
                self_attn, self.d_model, self.n_heads, self.mix
            )

            # Cross-attention (without mask)
            cross_attn = self._full_attention(False, self.factor, self.dropout, False)
            cross_attention_layer = self._attention_layer(
                cross_attn, self.d_model, self.n_heads, False
            )

            decoder_layer = self._decoder_layer(
                self_attention_layer,
                cross_attention_layer,
                self.d_model,
                self.d_ff,
                self.dropout,
                self.activation,
            )
            decoder_layers.append(decoder_layer)

        # Normalization layer
        norm_layer = tf.keras.layers.LayerNormalization()

        def decoder_function(inputs, training=None):
            x, cross = inputs
            x_mask = None  # Can be added as parameter if needed
            cross_mask = None  # Can be added as parameter if needed

            for layer in decoder_layers:
                x = layer(
                    x, cross, x_mask=x_mask, cross_mask=cross_mask, training=training
                )

            if norm_layer is not None:
                x = norm_layer(x, training=training)

            return x

        return decoder_function

    def _prob_attention(self, mask_flag, factor, attention_dropout, output_attention):
        """Create ProbSparse attention mechanism."""
        from math import sqrt

        import numpy as np
        import tensorflow as tf

        dropout_layer = tf.keras.layers.Dropout(attention_dropout)

        def _prob_QK(Q, K, sample_k, n_top):
            # Q [B, H, L, D]
            B, H, L_K, E = (
                tf.shape(K)[0],
                tf.shape(K)[1],
                tf.shape(K)[2],
                tf.shape(K)[3],
            )
            L_Q = tf.shape(Q)[2]

            # calculate the sampled Q_K
            K_expand = tf.expand_dims(K, axis=-3)  # [B, H, 1, L_K, E]
            K_expand = tf.tile(K_expand, [1, 1, L_Q, 1, 1])  # [B, H, L_Q, L_K, E]

            # Generate random indices for sampling
            index_sample = tf.random.uniform(
                (L_Q, sample_k), maxval=L_K, dtype=tf.int32
            )

            # Create indices for gathering
            batch_indices = tf.range(B)[:, None, None, None, None]
            head_indices = tf.range(H)[None, :, None, None, None]
            query_indices = tf.range(L_Q)[None, None, :, None, None]
            sample_indices = index_sample[None, None, :, :, None]

            # Gather K_sample
            gather_indices = tf.concat(
                [
                    tf.broadcast_to(batch_indices, [B, H, L_Q, sample_k, 1]),
                    tf.broadcast_to(head_indices, [B, H, L_Q, sample_k, 1]),
                    tf.broadcast_to(query_indices, [B, H, L_Q, sample_k, 1]),
                    tf.broadcast_to(sample_indices, [B, H, L_Q, sample_k, 1]),
                ],
                axis=-1,
            )

            K_sample = tf.gather_nd(
                K_expand, gather_indices
            )  # [B, H, L_Q, sample_k, E]

            # Calculate Q_K_sample
            Q_expanded = tf.expand_dims(Q, axis=-2)  # [B, H, L_Q, 1, E]
            Q_K_sample = tf.matmul(
                Q_expanded, K_sample, transpose_b=True
            )  # [B, H, L_Q, 1, sample_k]
            Q_K_sample = tf.squeeze(Q_K_sample, axis=-2)  # [B, H, L_Q, sample_k]

            # find the Top_k query with sparsity measurement
            M = tf.reduce_max(Q_K_sample, axis=-1) - tf.reduce_sum(
                Q_K_sample, axis=-1
            ) / tf.cast(L_K, tf.float32)
            M_top = tf.nn.top_k(M, k=n_top, sorted=False).indices

            # use the reduced Q to calculate Q_K
            batch_idx = tf.range(B)[:, None, None]
            head_idx = tf.range(H)[None, :, None]

            gather_indices_q = tf.stack(
                [
                    tf.broadcast_to(batch_idx, tf.shape(M_top)),
                    tf.broadcast_to(head_idx, tf.shape(M_top)),
                    M_top,
                ],
                axis=-1,
            )

            Q_reduce = tf.gather_nd(Q, gather_indices_q)  # [B, H, n_top, E]
            Q_K = tf.matmul(Q_reduce, K, transpose_b=True)  # [B, H, n_top, L_K]

            return Q_K, M_top

        def _get_initial_context(V, L_Q):
            B, H, L_V, D = (
                tf.shape(V)[0],
                tf.shape(V)[1],
                tf.shape(V)[2],
                tf.shape(V)[3],
            )

            if not mask_flag:
                V_sum = tf.reduce_mean(V, axis=-2)  # [B, H, D]
                context = tf.expand_dims(V_sum, axis=-2)  # [B, H, 1, D]
                context = tf.tile(context, [1, 1, L_Q, 1])  # [B, H, L_Q, D]
            else:
                # For masked case, L_Q should equal L_V
                context = tf.cumsum(V, axis=-2)

            return context

        def _prob_mask(B, H, L, index, scores):
            # Create upper triangular mask (excluding diagonal)
            L_scores = tf.shape(scores)[-1]
            _mask = tf.linalg.band_part(tf.ones((L, L_scores), dtype=tf.bool), 0, -1)
            _mask = tf.logical_not(_mask)  # Upper triangular without diagonal

            # Expand mask for batch and head dimensions
            _mask_ex = tf.tile(
                tf.expand_dims(tf.expand_dims(_mask, 0), 0), [B, H, 1, 1]
            )

            # Gather mask at specified indices
            batch_idx = tf.range(B)[:, None, None]
            head_idx = tf.range(H)[None, :, None]

            gather_indices = tf.stack(
                [
                    tf.broadcast_to(batch_idx, tf.shape(index)),
                    tf.broadcast_to(head_idx, tf.shape(index)),
                    index,
                ],
                axis=-1,
            )

            indicator = tf.gather_nd(_mask_ex, gather_indices)
            return indicator

        def _update_context(context_in, V, scores, index, L_Q, attn_mask):
            B, H, L_V, D = (
                tf.shape(V)[0],
                tf.shape(V)[1],
                tf.shape(V)[2],
                tf.shape(V)[3],
            )

            if mask_flag:
                attn_mask = _prob_mask(B, H, L_Q, index, scores)
                scores = tf.where(
                    attn_mask, tf.fill(tf.shape(scores), float("-inf")), scores
                )

            attn = tf.nn.softmax(scores, axis=-1)

            # Calculate attention-weighted values
            attn_V = tf.matmul(attn, V)  # [B, H, n_top, D]

            # Update context_in at specified indices
            batch_idx = tf.range(B)[:, None, None]
            head_idx = tf.range(H)[None, :, None]

            update_indices = tf.stack(
                [
                    tf.broadcast_to(batch_idx, tf.shape(index)),
                    tf.broadcast_to(head_idx, tf.shape(index)),
                    index,
                ],
                axis=-1,
            )

            context_in = tf.tensor_scatter_nd_update(context_in, update_indices, attn_V)

            if output_attention:
                # Initialize full attention matrix
                attns = tf.ones([B, H, L_V, L_V], dtype=attn.dtype) / tf.cast(
                    L_V, attn.dtype
                )
                attns = tf.tensor_scatter_nd_update(attns, update_indices, attn)
                return context_in, attns
            else:
                return context_in, None

        def prob_attention_function(
            queries, keys, values, attn_mask=None, training=None
        ):
            B, L_Q, H, D = (
                tf.shape(queries)[0],
                tf.shape(queries)[1],
                tf.shape(queries)[2],
                tf.shape(queries)[3],
            )
            L_K = tf.shape(keys)[1]

            # Transpose to [B, H, L, D] format
            queries = tf.transpose(queries, perm=[0, 2, 1, 3])
            keys = tf.transpose(keys, perm=[0, 2, 1, 3])
            values = tf.transpose(values, perm=[0, 2, 1, 3])

            # Calculate sampling parameters
            U_part = int(factor * np.ceil(np.log(L_K)))
            u = int(factor * np.ceil(np.log(L_Q)))

            U_part = min(U_part, L_K)
            u = min(u, L_Q)

            # Get top-k scores and indices
            scores_top, index = _prob_QK(queries, keys, sample_k=U_part, n_top=u)

            # Apply scale factor
            scale = 1.0 / sqrt(D)
            if scale is not None:
                scores_top = scores_top * scale

            # Get initial context and update with top-k queries
            context = _get_initial_context(values, L_Q)
            context, attn = _update_context(
                context, values, scores_top, index, L_Q, attn_mask
            )

            # Transpose back to [B, L, H, D] format
            context = tf.transpose(context, perm=[0, 2, 1, 3])

            return context, attn

        return prob_attention_function

    def _full_attention(self, mask_flag, factor, attention_dropout, output_attention):
        """Create full attention mechanism."""
        import numpy as np
        import tensorflow as tf

        dropout_layer = tf.keras.layers.Dropout(attention_dropout)

        def _triangular_causal_mask(B, L):
            """Create triangular causal mask for attention."""
            mask_shape = [B, 1, L, L]
            # Create upper triangular mask (excluding diagonal)
            mask = tf.linalg.band_part(tf.ones(mask_shape, dtype=tf.bool), 0, -1)
            mask = tf.logical_not(tf.linalg.band_part(mask, 0, 0))  # Remove diagonal
            return mask

        def full_attention_function(
            queries, keys, values, attn_mask=None, training=None
        ):
            # Get shapes
            B = tf.shape(queries)[0]
            L = tf.shape(queries)[1]
            H = tf.shape(queries)[2]
            E = tf.shape(queries)[3]
            S = tf.shape(keys)[1]
            D = tf.shape(values)[3]

            # Calculate scale
            scale = 1.0 / tf.math.sqrt(tf.cast(E, tf.float32))

            # Compute attention scores: "blhe,bshe->bhls"
            scores = tf.einsum("blhe,bshe->bhls", queries, keys)

            if mask_flag:
                if attn_mask is None:
                    attn_mask = _triangular_causal_mask(B, L)
                else:
                    # If attn_mask is provided, use its mask attribute if it's an object
                    if hasattr(attn_mask, "mask"):
                        attn_mask = attn_mask.mask

                # Apply mask by setting masked positions to -inf
                scores = tf.where(
                    attn_mask,
                    tf.fill(tf.shape(scores), tf.constant(-np.inf, dtype=scores.dtype)),
                    scores,
                )

            # Apply scale and softmax
            A = tf.nn.softmax(scale * scores, axis=-1)

            # Apply dropout
            A = dropout_layer(A, training=training)

            # Compute output: "bhls,bshd->blhd"
            V = tf.einsum("bhls,bshd->blhd", A, values)

            if output_attention:
                return V, A
            else:
                return V, None

        return full_attention_function

    def _attention_layer(self, attention, d_model, n_heads, mix):
        """Create attention layer wrapper."""
        import tensorflow as tf

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        # Linear projection layers for Q, K, V
        query_dense = tf.keras.layers.Dense(d_model)
        key_dense = tf.keras.layers.Dense(d_model)
        value_dense = tf.keras.layers.Dense(d_model)

        # Output projection
        out_projection = tf.keras.layers.Dense(d_model)

        def attention_layer_function(
            queries, keys, values, attn_mask=None, training=None
        ):
            B, L, _ = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(queries)[2]
            S = tf.shape(keys)[1]
            H = n_heads

            # Linear projections in batch from d_model => h x d_k
            Q = query_dense(queries)
            K = key_dense(keys)
            V = value_dense(values)

            # Reshape to (B, L, H, d_k) and transpose to (B, H, L, d_k)
            Q = tf.reshape(Q, [B, L, H, d_keys])
            K = tf.reshape(K, [B, S, H, d_keys])
            V = tf.reshape(V, [B, S, H, d_values])

            Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, H, L, d_k)
            K = tf.transpose(K, [0, 2, 1, 3])  # (B, H, S, d_k)
            V = tf.transpose(V, [0, 2, 1, 3])  # (B, H, S, d_v)

            # Apply attention function
            out, attn = attention(Q, K, V, attn_mask=attn_mask, training=training)

            # Concatenate heads and put through final linear layer
            # out shape: (B, H, L, d_v) -> (B, L, H, d_v) -> (B, L, H*d_v)
            out = tf.transpose(out, [0, 2, 1, 3])
            out = tf.reshape(out, [B, L, H * d_values])

            # Apply mix transformation if needed
            if mix:
                # Reshape to (B, L, H, d_values) then transpose to (B, H, L, d_values)
                out = tf.reshape(out, [B, L, H, d_values])
                out = tf.transpose(out, [0, 2, 1, 3])
                out = tf.reshape(out, [B, L, H * d_values])

            # Final output projection
            out = out_projection(out)

            return out, attn

        return attention_layer_function

    def _encoder_layer(self, attention_layer, d_model, d_ff, dropout, activation):
        """Create single encoder layer."""
        import tensorflow as tf

        d_ff = d_ff or 4 * d_model

        # Conv1D layers for feed-forward network
        conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1)
        conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)

        # Layer normalization
        norm1 = tf.keras.layers.LayerNormalization()
        norm2 = tf.keras.layers.LayerNormalization()

        # Dropout
        dropout_layer = tf.keras.layers.Dropout(dropout)

        # Activation function
        if activation == "relu":
            activation_fn = tf.nn.relu
        else:
            activation_fn = tf.nn.gelu

        def encoder_layer_function(x, attn_mask=None, training=None):
            # Self-attention with residual connection
            new_x, attn = attention_layer(
                x, x, x, attn_mask=attn_mask, training=training
            )
            x = x + dropout_layer(new_x, training=training)
            y = x = norm1(x, training=training)

            # Feed-forward network with residual connection
            y = conv1(y)
            y = dropout_layer(activation_fn(y), training=training)
            y = conv2(y)
            y = dropout_layer(y, training=training)

            return norm2(x + y, training=training), attn

        return encoder_layer_function

    def _decoder_layer(
        self, self_attention, cross_attention, d_model, d_ff, dropout, activation
    ):
        """Create single decoder layer."""
        import tensorflow as tf

        d_ff = d_ff or 4 * d_model

        # Conv1D layers equivalent to PyTorch's Conv1d
        conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1)
        conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)

        # Layer normalization
        norm1 = tf.keras.layers.LayerNormalization()
        norm2 = tf.keras.layers.LayerNormalization()
        norm3 = tf.keras.layers.LayerNormalization()

        # Dropout
        dropout_layer = tf.keras.layers.Dropout(dropout)

        # Activation function
        if activation == "relu":
            activation_fn = tf.nn.relu
        else:
            activation_fn = tf.nn.gelu

        def decoder_layer_function(
            x, cross, x_mask=None, cross_mask=None, training=None
        ):
            # Self-attention with residual connection
            self_attn_out = self_attention(
                x, x, x, attn_mask=x_mask, training=training
            )[0]
            x = x + dropout_layer(self_attn_out, training=training)
            x = norm1(x, training=training)

            # Cross-attention with residual connection
            cross_attn_out = cross_attention(
                x, cross, cross, attn_mask=cross_mask, training=training
            )[0]
            x = x + dropout_layer(cross_attn_out, training=training)
            y = x = norm2(x, training=training)

            # Feed-forward network with residual connection
            y = conv1(y)
            y = dropout_layer(activation_fn(y), training=training)
            y = conv2(y)
            y = dropout_layer(y, training=training)

            return norm3(x + y, training=training)

        return decoder_layer_function

    def _conv_layer(self, d_model):
        """Create convolution layer for distilling."""
        import tensorflow as tf

        # TensorFlow doesn't have direct circular padding, using 'same' padding
        downConv = tf.keras.layers.Conv1D(
            filters=d_model, kernel_size=3, padding="same", activation=None
        )
        norm = tf.keras.layers.BatchNormalization()
        activation = tf.keras.layers.ELU()
        maxPool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")

        def conv_layer_function(x, training=None):
            # x shape: [B, L, D] -> Conv1D expects [B, L, C]
            x = downConv(x)
            x = norm(x, training=training)
            x = activation(x)
            x = maxPool(x)
            return x

        return conv_layer_function
