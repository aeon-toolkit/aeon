"""Abstract base class for deep learning networks."""

__maintainer__ = []

from abc import ABC, abstractmethod

from aeon.base import BaseObject
from aeon.utils.validation._dependencies import _check_estimator_deps


class BaseDeepNetwork(BaseObject, ABC):
    """Abstract base class for deep learning networks."""

    def __init__(self):
        super().__init__()
        _check_estimator_deps(self)

    _tags = {
        "python_dependencies": "tensorflow",
        "python_version": "<3.12",
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
