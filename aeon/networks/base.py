"""Abstract base class for deep learning networks."""

__maintainer__ = ["hadifawaz1999"]

from abc import ABC, abstractmethod

import numpy as np

from aeon.utils.repr import get_unchanged_and_required_params_as_str
from aeon.utils.validation._dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)


class BaseDeepLearningNetwork(ABC):
    """Abstract base class for deep learning networks."""

    _config = {
        "python_dependencies": "tensorflow",
        "python_version": "<3.14",
    }

    @abstractmethod
    def __init__(self):
        _check_soft_dependencies(self._config["python_dependencies"])
        _check_python_version(self._config["python_version"])
        super().__init__()

    def __repr__(self):
        """Format str output like scikit-learn estimators."""
        changed_params = get_unchanged_and_required_params_as_str(self)
        return f"{self.__class__.__name__}({changed_params})"

    @staticmethod
    def _check_layer_param(
        depth: int,
        param: list | int | float | str = None,
        param_name: str = "None",
        default=None,
        allow_none: bool = False,
        same_as: str = None,
    ):
        """
        Check and convert a network parameter to a list of length n_layers.

        Parameters
        ----------
        depth : int
            The depth of the network (generally self.n_layers).
        param_name : str
            The name of the parameter to check (used for error messages).
        param : list | int | float | str
            The parameter to check. Can be a list or a single value of any type.
        default : list | int | float | str,
            The default value to use if the parameter is None.
        allow_none: bool = False
            Whether to accept None as a valid value.
        same_as: str = None
            The label to use to indicate "the depth" in error messages.
            eg. "number of layers" or "number of blocks".
        """
        same_as = "number of layers" if same_as is None else same_as

        if (default is None) and (not allow_none):
            raise ValueError(
                f"Add default value for parameter {param_name} or set allow_none=True."
            )

        if param is None:
            if allow_none:
                return [None] * depth
            if default is None:
                raise ValueError(
                    f"Parameter {param_name} is None, but no default value is provided."
                )
            param = default

        if isinstance(param, list):
            if len(param) != depth:
                raise ValueError(
                    f"Number of {param_name} ({len(param)}) should be"
                    f" the same as {same_as} ({depth})."
                )
            return param
        else:
            return [param] * depth

    @abstractmethod
    def _check_params(self):
        """Check and convert parameters to lists of length n_layers before building."""
        ...

    @abstractmethod
    def build_base_graph(self, x):
        """Construct the network graph without input and output layers.

        Used to embed the network in any larger network

        Parameters
        ----------
        x : tf.Tensor
            The input tensor to the network. Can be
            any tensorflow layer, generally an Input layer.

        Returns
        -------
        x : tf.Tensor
            The last layer of the network, generally used as
            input for the final output layer of the network.
        """
        ...

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
          shape = (n_timepoints (m), n_channels (d)), the shape of the data fed
          into the input layer.

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)
        x = self.build_base_graph(input_layer)
        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap_layer


class BaseDeepAENetwork(BaseDeepLearningNetwork):
    """Abstract base class for deep autoencoder networks."""

    def __init__(self, latent_space_dim, temporal_latent_space, repeated_latent_space):

        self._enc_out = None
        self._dec_in = None
        self.latent_space_dim = latent_space_dim
        self.temporal_latent_space = temporal_latent_space
        self.repeated_latent_space = repeated_latent_space

        if self.temporal_latent_space and self.repeated_latent_space:
            raise ValueError(
                "temporal_latent_space and repeated_latent_space cannot both be True."
            )

    def _build_latent_graph(self, x):
        import tensorflow as tf

        enc_out_shape = x.shape[1:]

        if self.repeated_latent_space:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(self.latent_space_dim)(x)
        elif not self.temporal_latent_space:
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(self.latent_space_dim)(x)
        else:
            x = tf.keras.layers.Conv1D(
                filters=self.latent_space_dim,
                kernel_size=1,
                strides=self._strides[-1],
                padding=self._padding[-1],
                dilation_rate=self._dilation_rate[-1],
                use_bias=self._use_bias[-1],
            )(x)

        self._enc_out = x
        self._dec_in = x

        if self.repeated_latent_space:
            x = tf.keras.layers.RepeatVector(enc_out_shape[0])(x)

        elif not self.temporal_latent_space:
            decoder_units = int(np.prod(enc_out_shape))

            x = tf.keras.layers.Dense(units=decoder_units)(x)
            x = tf.keras.layers.Reshape(target_shape=enc_out_shape)(x)
        return x

    @abstractmethod
    def _build_encoder_graph(self, x):
        """Construct the encoder graph of the autoencoder."""
        ...

    @abstractmethod
    def _build_decoder_graph(self, x):
        """Construct the decoder graph of the autoencoder."""
        ...

    def _build_projection_graph(self, x):
        import tensorflow as tf

        return tf.keras.layers.Conv1DTranspose(
            filters=self._input_shape[-1],
            kernel_size=1,
            use_bias=self._use_bias[0],
        )(x)

    def build_base_graph(self, x):
        """Construct the network graph without input and output layers.

        Used to embed the network in any larger network

        Parameters
        ----------
        x : tf.Tensor
            The input tensor to the network. Can be
            any tensorflow layer, generally an Input layer.

        Returns
        -------
        x : tf.Tensor
            The last layer of the network, generally used as
            input for the final output layer of the network.
        """
        self._check_params()
        x = self._build_encoder_graph(x)
        x = self._build_latent_graph(x)
        x = self._build_decoder_graph(x)
        x = self._build_projection_graph(x)
        return x

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

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

        self._enc_in = tf.keras.layers.Input(input_shape)
        self._dec_out = self.build_base_graph(self._enc_in)

        encoder = tf.keras.Model(
            inputs=self._enc_in, outputs=self._enc_out, name="encoder"
        )
        decoder = tf.keras.Model(
            inputs=self._dec_in, outputs=self._dec_out, name="decoder"
        )
        return encoder, decoder
