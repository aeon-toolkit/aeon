"""Full Attention, ProbSparseAttention and Attention Layer."""

from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.layers import Dropout, Layer


class KerasProbAttention(Layer):
    """Keras implementation of ProbSparse Attention mechanism for Informer."""

    def __init__(
        self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, **kwargs
    ):
        """Initialize KerasProbAttention layer."""
        super().__init__(**kwargs)
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.attention_dropout = attention_dropout
        self.dropout = Dropout(attention_dropout)

    def build(self, input_shape):
        """Build the layer."""
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute output shape for the layer."""
        # Return the same shape as queries input
        return input_shape[0]  # queries shape

    def compute_output_spec(self, input_spec):
        """Compute output spec for the layer."""
        return input_spec[0]  # Return queries spec

    def _prob_QK(self, Q, K, sample_k, n_top):
        """Compute probabilistic QK with fixed dimension handling."""
        B, H, L, _ = tf.shape(Q)[0], tf.shape(Q)[1], tf.shape(Q)[2], tf.shape(Q)[3]
        S = tf.shape(K)[2]

        # Ensure sample_k doesn't exceed available dimensions
        sample_k = tf.minimum(sample_k, L)
        n_top = tf.minimum(n_top, S)  # Ensure n_top doesn't exceed sequence length

        # Expand K for sampling
        K_expand = tf.expand_dims(K, axis=2)  # [B, H, 1, L, E]
        K_expand = tf.tile(K_expand, [1, 1, S, 1, 1])  # [B, H, S, L, E]

        # Generate random indices - ensure they're within bounds
        indx_q_seq = tf.random.uniform([S], maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform([sample_k], maxval=L, dtype=tf.int32)

        # Gather operations for sampling
        indices_s = tf.range(S)
        K_sample = tf.gather(K_expand, indices_s, axis=2)
        K_sample = tf.gather(K_sample, indx_q_seq, axis=2)
        K_sample = tf.gather(K_sample, indx_k_seq, axis=3)

        # Matrix multiplication for Q_K_sample
        Q_expanded = tf.expand_dims(Q, axis=-2)  # [B, H, S, 1, E]
        K_sample_transposed = tf.transpose(K_sample, perm=[0, 1, 2, 4, 3])
        Q_K_sample = tf.squeeze(tf.matmul(Q_expanded, K_sample_transposed), axis=-2)

        # Sparsity measurement calculation
        M_max = tf.reduce_max(Q_K_sample, axis=-1)
        M_mean = tf.reduce_sum(Q_K_sample, axis=-1) / tf.cast(sample_k, tf.float32)
        M = M_max - M_mean

        # Top-k selection with dynamic k
        actual_k = tf.minimum(n_top, tf.shape(M)[-1])
        _, M_top = tf.nn.top_k(M, k=actual_k, sorted=False)

        # Create indices for gather_nd
        batch_range = tf.range(B)
        head_range = tf.range(H)
        batch_indices = tf.tile(
            tf.expand_dims(tf.expand_dims(batch_range, 1), 2), [1, H, actual_k]
        )

        head_indices = tf.tile(
            tf.expand_dims(tf.expand_dims(head_range, 0), 2), [B, 1, actual_k]
        )

        # Stack indices for gather_nd
        idx = tf.stack([batch_indices, head_indices, M_top], axis=-1)

        # Reduce Q and calculate final Q_K
        Q_reduce = tf.gather_nd(Q, idx)
        K_transposed = tf.transpose(K, perm=[0, 1, 3, 2])
        Q_K = tf.matmul(Q_reduce, K_transposed)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """Get initial context using Keras-compatible operations."""
        if not self.mask_flag:
            # Sum reduction and broadcasting
            V_sum = tf.reduce_sum(V, axis=-2)  # [B, H, D]
            V_sum_expanded = tf.expand_dims(V_sum, axis=-2)  # [B, H, 1, D]
            context = tf.tile(V_sum_expanded, [1, 1, L_Q, 1])  # [B, H, L_Q, D]
        else:
            # Cumulative sum for masked attention
            context = tf.cumsum(V, axis=-2)

        return context

    def _create_prob_mask(self, B, H, L, index, scores):
        """Create probability mask for tf.where compatibility."""
        # Create base mask with ones
        _mask = tf.ones((L, tf.shape(scores)[-1]), dtype=tf.float32)

        # Create upper triangular matrix (including diagonal)
        mask_a = tf.linalg.band_part(
            _mask, 0, -1
        )  # Upper triangular matrix of 0s and 1s

        # Create diagonal matrix
        mask_b = tf.linalg.band_part(_mask, 0, 0)  # Diagonal matrix of 0s and 1s

        # Subtract diagonal from upper triangular to get strict upper triangular
        _mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        # Broadcast to [B, H, L, scores.shape[-1]]
        _mask_ex = tf.broadcast_to(_mask, [B, H, L, tf.shape(scores)[-1]])

        # Create indexing tensors
        batch_indices = tf.range(B)[:, None, None]
        head_indices = tf.range(H)[None, :, None]

        # Extract indicator using advanced indexing
        indicator = tf.gather_nd(
            _mask_ex,
            tf.stack(
                [
                    tf.broadcast_to(batch_indices, tf.shape(index)),
                    tf.broadcast_to(head_indices, tf.shape(index)),
                    index,
                ],
                axis=-1,
            ),
        )

        # Reshape to match scores shape
        prob_mask_float = tf.reshape(indicator, tf.shape(scores))

        # **KEY FIX**: Convert to boolean tensor
        prob_mask_bool = tf.cast(prob_mask_float, tf.bool)

        return prob_mask_bool

    def _update_context(self, context_in, V, scores, index, L_Q):
        """Update context using Keras-compatible operations."""
        if self.mask_flag:
            # Apply simple masking
            attn_mask = self._create_prob_mask(
                tf.shape(V)[0], tf.shape(V)[1], L_Q, index, scores
            )

            # Apply mask with large negative value
            large_neg = -1e9
            mask_value = tf.where(attn_mask, 0.0, large_neg)
            scores = scores + mask_value

        # Softmax activation
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn)

        # Create indices for scatter update
        B, H = tf.shape(V)[0], tf.shape(V)[1]
        index_shape = tf.shape(index)[-1]

        batch_indices = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(B), 1), 2), [1, H, index_shape]
        )

        head_indices = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(H), 0), 2), [B, 1, index_shape]
        )

        idx = tf.stack([batch_indices, head_indices, index], axis=-1)

        # Matrix multiplication and scatter update
        attn_V = tf.matmul(attn, V)
        context_updated = tf.tensor_scatter_nd_update(context_in, idx, attn_V)

        return context_updated

    def call(self, inputs, attention_mask=None, training=None):
        """Run forward pass with fixed tensor operations."""
        queries, keys, values = inputs

        # Get shapes
        # B = tf.shape(queries)[0]
        L = tf.shape(queries)[1]  # sequence length
        # H = tf.shape(queries)[2]  # number of heads
        D = tf.shape(queries)[3]  # dimension per head
        S = tf.shape(keys)[1]  # source sequence length

        # Reshape tensors - transpose to [B, H, L, D]
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])  # [B, H, L, D]
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])  # [B, H, S, D]
        values = tf.transpose(values, perm=[0, 2, 1, 3])  # [B, H, S, D]

        # Calculate sampling parameters with bounds checking
        # Use tf.py_function to handle numpy operations safely
        def safe_log_calc(seq_len, factor):
            if hasattr(seq_len, "numpy"):
                return int(factor * np.ceil(np.log(max(seq_len.numpy(), 2))))
            else:
                return int(factor * np.ceil(np.log(20)))  # fallback

        U = tf.py_function(
            func=lambda: safe_log_calc(S, self.factor), inp=[], Tout=tf.int32
        )

        u = tf.py_function(
            func=lambda: safe_log_calc(L, self.factor), inp=[], Tout=tf.int32
        )

        # Ensure U and u are within reasonable bounds
        U = tf.minimum(U, S)  # Can't select more than available
        u = tf.minimum(u, L)

        # Probabilistic QK computation
        scores_top, index = self._prob_QK(queries, keys, u, U)

        # Apply scale factor
        scale = self.scale or (1.0 / tf.sqrt(tf.cast(D, tf.float32)))
        scores_top = scores_top * scale

        # Get initial context
        context = self._get_initial_context(values, L)

        # Update context with selected queries
        context = self._update_context(context, values, scores_top, index, L)

        # Transpose back to original format [B, L, H, D]
        context = tf.transpose(context, perm=[0, 2, 1, 3])

        return context

    def get_config(self):
        """Return the config of the layer."""
        config = super().get_config()
        config.update(
            {
                "mask_flag": self.mask_flag,
                "factor": self.factor,
                "scale": self.scale,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from config."""
        return cls(**config)


class AttentionLayer(Layer):
    """Keras multi-head attention layer using a custom attention mechanism."""

    def __init__(
        self, attention, d_model, n_heads, d_keys=None, d_values=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)
        self.d_model = d_model
        self.n_heads = n_heads

        # Store the attention mechanism
        self.inner_attention = attention

        # Projection layers
        self.query_projection = tf.keras.layers.Dense(
            self.d_keys * n_heads, name="query_proj"
        )

        self.key_projection = tf.keras.layers.Dense(
            self.d_keys * n_heads, name="key_proj"
        )

        self.value_projection = tf.keras.layers.Dense(
            self.d_values * n_heads, name="value_proj"
        )

        self.out_projection = tf.keras.layers.Dense(d_model, name="output_proj")

    def build(self, input_shape):
        """Build the layer."""
        # Build the projection layers
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute output shape for the layer."""
        # Output shape is same as queries input shape but with d_model as last dimension
        batch_size, seq_length, _ = input_shape[0]
        return (batch_size, seq_length, self.d_model)

    def call(self, inputs, attn_mask=None, training=None):
        """Run forward pass for the attention layer."""
        queries, keys, values = inputs

        # Get batch size and sequence lengths dynamically
        B = tf.shape(queries)[0]
        L = tf.shape(queries)[1]  # target sequence length
        S = tf.shape(keys)[1]  # source sequence length
        H = self.n_heads

        # Apply projections
        queries_proj = self.query_projection(queries)  # [B, L, d_keys * n_heads]
        keys_proj = self.key_projection(keys)  # [B, S, d_keys * n_heads]
        values_proj = self.value_projection(values)  # [B, S, d_values * n_heads]

        # Reshape to multi-head format: [B, L/S, H, d_keys/d_values]
        queries_reshaped = tf.reshape(queries_proj, (B, L, H, self.d_keys))
        keys_reshaped = tf.reshape(keys_proj, (B, S, H, self.d_keys))
        values_reshaped = tf.reshape(values_proj, (B, S, H, self.d_values))

        # Apply inner attention mechanism
        attention_output = self.inner_attention(
            [queries_reshaped, keys_reshaped, values_reshaped],
            attention_mask=attn_mask,
            training=training,
        )

        # Reshape attention output back to [B, L, H * d_values]
        attention_flattened = tf.reshape(attention_output, (B, L, H * self.d_values))

        # Final output projection
        output = self.out_projection(attention_flattened)

        return output

    def get_config(self):
        """Return the config of the layer."""
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "d_keys": self.d_keys,
                "d_values": self.d_values,
            }
        )
        return config
