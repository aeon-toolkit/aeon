"""Abstract base class for deep learning networks."""

__maintainer__ = []

from abc import ABC, abstractmethod

from aeon.utils.validation._dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)


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
