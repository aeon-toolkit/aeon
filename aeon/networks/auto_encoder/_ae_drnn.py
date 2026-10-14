"""Auto-Encoder based Dilated Recurrent Neural Networks (DRNN)."""

__maintainer__ = ["aadya940", "hadifawaz1999"]

import numpy as np

from aeon.networks.base import BaseDeepAENetwork
from aeon.typing import LATENT_SPACE
from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class _TensorDilation(tf.keras.layers.Layer):
        """A layer for dilation of a tensorflow tensor."""

        def __init__(self, dilation_rate, **kwargs):
            super().__init__(**kwargs)
            self._dilation_rate = dilation_rate

        def call(self, inputs):
            return inputs[:, :: self._dilation_rate, :]

        def get_config(self):
            config = super().get_config()
            config.update({"dilation_rate": self._dilation_rate})
            return config

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class DRNN_BidirectionalGRU(tf.keras.layers.Layer):

        def __init__(self, nunits, activation="tanh", **kwargs):
            super().__init__(**kwargs)

            self.nunits = nunits
            self.activation = activation

            self.gru = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    nunits,
                    activation=activation,
                    return_sequences=True,
                    return_state=True,
                )
            )

        def call(self, inputs):
            output, forward_h, backward_h = self.gru(inputs)

            final_state = tf.keras.layers.Concatenate()([forward_h, backward_h])

            return output, final_state

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "nunits": self.nunits,
                    "activation": tf.keras.activations.serialize(
                        tf.keras.activations.get(self.activation)
                    ),
                }
            )
            return config


class AEDRNNNetwork(BaseDeepAENetwork):
    """Auto-Encoder based Dilated Recurrent Neural Networks (DRNN).

    Parameters
    ----------
    latent_space_dim : int, default = 128
        Dimensionality of the latent space.
    latent_space_type : LATENT_SPACE, default = LATENT_SPACE.REPEATED
        Type of latent space to use. Options are:
        - LATENT_SPACE.FLAT: The latent space is a flattened vector.
        - LATENT_SPACE.TIME: The latent space is a time series.
        - LATENT_SPACE.REPEATED: The latent space is a repeated vector.
    n_layers_encoder : int, default = 3
        Number of GRU layers in the encoder.
    n_layers_decoder : int, default = 1
        Number of GRU layers in the decoder.
    dilation_rate_encoder : Union[int, List[int]], default = None
        List of dilation rates for each layer of the encoder.
        If None, default = powers of 2 up to `n_stacked`.
    dilation_rate_decoder : Union[int, List[int]], default = None
        List of dilation rates for each layer of the decoder.
        If None, default to a list of ones.
    activation_encoder : Union[str, List[str]], default="relu"
        Activation function to use in the GRU layers.
    activation_decoder : Union[str, List[str]], default="relu"
        Activation function of the single GRU layer in the decoder.
    n_units_encoder : List[int], default="None"
        Number of units in each GRU layer of the encoder, by default None.
        If None, default to [100, 50, 50].
    n_units_decoder : List[int], default="None"
        Number of units in each GRU layer of the decoder, by default None.
        If None, default to two times sum of units of the encoder.
    """

    _config = {
        **BaseDeepAENetwork._config,
        "structure": "auto-encoder",
    }

    def __init__(
        self,
        latent_space_dim=128,
        latent_space_type=LATENT_SPACE.REPEATED,
        n_layers_encoder=3,
        n_layers_decoder=1,
        dilation_rate_encoder=None,
        dilation_rate_decoder=1,
        activation_encoder="relu",
        activation_decoder="relu",
        n_units_encoder=None,
        n_units_decoder=None,
    ):
        super().__init__(latent_space_dim, latent_space_type)

        self.latent_space_dim = latent_space_dim
        self.latent_space_type = latent_space_type
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.dilation_rate_encoder = dilation_rate_encoder
        self.dilation_rate_decoder = dilation_rate_decoder
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.n_units_encoder = n_units_encoder
        self.n_units_decoder = n_units_decoder

    def _check_params(self):
        enc_l = self.n_layers_encoder
        dec_l = self.n_layers_decoder
        enc_act = self.activation_encoder
        dec_act = self.activation_decoder

        self._latent_space_type = LATENT_SPACE._check_param(
            self.latent_space_type,
        )
        default = [2**l for l in range(1, self.n_layers_encoder + 1)]
        self._dilation_rate_encoder = BaseDeepAENetwork._check_layer_param(
            enc_l, self.dilation_rate_encoder, "dilation rates for encoder", default
        )
        self._dilation_rate_decoder = BaseDeepAENetwork._check_layer_param(
            dec_l, self.dilation_rate_decoder, "dilation rates for decoder", default=1
        )
        self._activation_encoder = BaseDeepAENetwork._check_layer_param(
            enc_l, enc_act, "activation for encoder", allow_none=True
        )
        self._activation_decoder = BaseDeepAENetwork._check_layer_param(
            dec_l, dec_act, "activation for decoder", allow_none=True
        )
        default = [100] + [50 for _ in range(self.n_layers_encoder - 1)]
        self._n_units_encoder = BaseDeepAENetwork._check_layer_param(
            enc_l, self.n_units_encoder, "units for encoder", default
        )
        default = [sum(self._n_units_encoder) * 2 for _ in range(self.n_layers_decoder)]
        self._n_units_decoder = BaseDeepAENetwork._check_layer_param(
            dec_l, self.n_units_decoder, "units for decoder", default
        )
        # add default value for _use_bias = [True]
        # for compatibility with _build_projection_graph
        self._use_bias = BaseDeepAENetwork._check_layer_param(
            1, param_name="use_bias", default=True
        )

    def _build_encoder_graph(self, x):
        _finals = []

        for i in range(self.n_layers_encoder):
            x, final = DRNN_BidirectionalGRU(
                self._n_units_encoder[i], activation=self._activation_encoder[i]
            )(x)
            if i < self.n_layers_encoder - 1:
                x = _TensorDilation(self._dilation_rate_encoder[i])(x)
            _finals.append(final)

        finals = tf.keras.layers.Concatenate()(_finals)
        return x, finals

    def _build_flat_latent_graph(self, x):
        import tensorflow as tf

        enc_out_shape = (self._input_shape[0], self._n_units_encoder[-1] * 2)
        decoder_units = int(np.prod(enc_out_shape))

        x = self._build_latent_projection(x[1])  # use final
        x = tf.keras.layers.Dense(units=decoder_units)(x)
        x = tf.keras.layers.Reshape(target_shape=enc_out_shape)(x)
        return x

    def _build_repeated_latent_graph(self, x):
        import tensorflow as tf

        x = self._build_latent_projection(x[1])  # use final
        x = tf.keras.layers.RepeatVector(self._input_shape[0])(x)
        return x

    def _build_time_latent_graph(self, x):
        return self._build_latent_projection(x[0])  # use x sequence

    def _build_decoder_graph(self, x):
        for i in range(self.n_layers_decoder):
            x = tf.keras.layers.GRU(
                self._n_units_decoder[i],
                return_sequences=True,
                activation=self._activation_decoder[i],
            )(x)
            if i < self.n_layers_decoder - 1:
                x = _TensorDilation(self._dilation_rate_decoder[i])(x)
        return x
