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
    def _resolve_param(param, param_name, default=None, allow_none=False):
        """Resolve a parameter using its default or None policy."""
        if param is not None:
            return param
        if allow_none:
            return None
        if default is None:
            raise ValueError(
                f"Parameter {param_name} is None, but no default value "
                "is provided. Set allow_none=True to accept None."
            )
        return default

    @staticmethod
    def _check_layer_param(
        depth: int,
        param: list | int | float | str | None,
        param_name: str,
        default=None,
        allow_none: bool = False,
        same_as: str = "number of layers",
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
        param = BaseDeepLearningNetwork._resolve_param(
            param, param_name, default, allow_none
        )

        if param is None:
            return [None] * depth

        if isinstance(param, list):
            if len(param) != depth:
                raise ValueError(
                    f"Number of {param_name} ({len(param)}) should be "
                    f"the same as {same_as} ({depth})."
                )
            return param

        return [param] * depth

    @staticmethod
    def _check_block_param(
        blocks: int,
        depth: int | list[int],
        param: list | int | float | str | None,
        param_name: str,
        default=None,
        allow_none: bool = False,
        block_name: str = "block",
        block_depth_name: str = "depth",
    ):
        """
        Check and convert a parameter to a list of list of length blocks and depth.

        Parameters
        ----------
        block : int
            The number of blocks.
        depth : int or list[int]
            The depth of the blocks.
        param : list[list] | list | int | float | str
            The parameter to check. Can be a list or a single value of any type.
        param_name : str
            The name of the parameter to check (used for error messages).
        default : list[list] | list | int | float | str,
            The default value to use if the parameter is None.
        block_name: str = "block"
            The label to use to indicate "the block" in error messages.
        block_depth_name: str = "elements"
            The label to use to indicate "the depth" in error messages.
        allow_none: bool = False
            Whether to accept None as a valid value.
        """
        depths = [depth] * blocks if isinstance(depth, int) else depth

        if len(depths) != blocks:
            raise ValueError(
                f"The number of {block_depth_name} per {block_name} "
                f"({len(depths)}) should be the same as the number of "
                f"{block_name} ({blocks})."
            )

        param = BaseDeepLearningNetwork._resolve_param(
            param, param_name, default, allow_none
        )

        if param is None:
            return [[None] * d for d in depths]

        # Nested list: parameters specified independently for each block.
        if isinstance(param, list) and param and isinstance(param[0], list):
            if len(param) != blocks:
                raise ValueError(
                    f"The length of {param_name} ({len(param)}) should be "
                    f"the same as the number of {block_name} ({blocks})."
                )

            return [
                BaseDeepLearningNetwork._check_layer_param(
                    depth=d,
                    param=values,
                    param_name=param_name,
                    same_as=(f"number of {block_depth_name} in {block_name} {i}"),
                )
                for i, (values, d) in enumerate(zip(param, depths))
            ]

        # Flat list: shared parameters require equal block depths.
        if isinstance(param, list):
            if depths and not all(d == depths[0] for d in depths):
                raise ValueError(
                    f"Every {block_name} should have the same number of "
                    f"{block_depth_name} if {param_name} is given as a "
                    f"1D list, but got {depths}."
                )

            values = BaseDeepLearningNetwork._check_layer_param(
                depth=depths[0] if depths else 0,
                param=param,
                param_name=param_name,
                same_as=f"number of {block_depth_name} per {block_name}",
            )
            return [values] * blocks

        # Scalar: expand independently to support different block depths.
        return [
            BaseDeepLearningNetwork._check_layer_param(
                depth=d,
                param=param,
                param_name=param_name,
            )
            for d in depths
        ]

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
