"""Abstract base class for deep learning networks."""

__maintainer__ = []

from abc import ABC, abstractmethod

from deprecated.sphinx import deprecated

from aeon.base import BaseObject
from aeon.utils.validation._dependencies import (
    _check_estimator_deps,
    _check_python_version,
    _check_soft_dependencies,
)


# TODO: remove v0.11.0
@deprecated(
    version="0.10.0",
    reason="BaseDeepNetwork will be removed in 0.11.0, use BaseDeepLearningNetwork "
    "instead. The new class does not inherit from BaseObject.",
    category=FutureWarning,
)
class BaseDeepNetwork(BaseObject, ABC):
    """Abstract base class for deep learning networks."""

    def __init__(self):
        super().__init__()
        _check_estimator_deps(self)

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.12",
        "structure": "encoder",
    }

    @abstractmethod
    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

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


class BaseDeepLearningNetwork(ABC):
    """Abstract base class for deep learning networks."""

    def __init__(self, soft_dependencies="tensorflow", python_version="<3.12"):
        _check_soft_dependencies(soft_dependencies)
        _check_python_version(python_version)
        super().__init__()

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.12",
        "structure": "encoder",
    }

    @abstractmethod
    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

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
