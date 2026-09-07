"""Abstract base class for deep learning networks."""

__maintainer__ = ["hadifawaz1999"]

from abc import ABC, abstractmethod

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
        n_layer: int,
        param_name: str,
        param: list | int | float | str,
        default=None,
        accept_none: bool = False,
        depth_label: str = None,
    ):
        """
        Check and convert a network parameter to a list of length n_layers.

        Parameters
        ----------
        n_layer : int
            The number of layers in the network (generally self.n_layers).
        param_name : str
            The name of the parameter to check (used for error messages).
        param : list | int | float | str
            The parameter to check. Can be a list or a single value of any type.
        default : list | int | float | str,
            The default value to use if the parameter is None.
        accept_none: bool = False
            Whether to accept None as a valid value.
        depth_label: str = None
            The label to use to indicate "the depth" in error messages.
            eg. "number of layers" or "number of blocks".
        """
        depth_label = "number of layers" if depth_label is None else depth_label

        if param is None:
            if accept_none:
                return [None] * n_layer
            if default is None:
                raise ValueError(
                    f"Parameter {param_name} is None, but no default value is provided."
                )
            param = default

        if isinstance(param, list):
            if len(param) != n_layer:
                raise ValueError(
                    f"Number of {param_name} {len(param)} should be"
                    f" the same as {depth_label} but is"
                    f" not: {n_layer}"
                )
            return param
        else:
            return [param] * n_layer

    @abstractmethod
    def _check_params(self):
        """Check and convert parameters to lists of length n_layers before building."""
        ...

    @abstractmethod
    def build_base_graph(self, x):
        """Construct the network graph without input and output layers.

        Used to embed the network in any larger network.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
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
