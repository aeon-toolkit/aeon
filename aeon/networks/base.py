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
        depth: int,
        param: list | int | float | str,
        param_name: str,
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
                    f"Number of {param_name} {len(param)} should be"
                    f" the same as {same_as} but is"
                    f" not: {depth}"
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
