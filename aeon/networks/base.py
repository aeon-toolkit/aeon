"""Abstract base class for deep learning networks."""

__author__ = ["hadifawaz1999", "Withington", "TonyBagnall"]

from abc import ABC, abstractmethod

from aeon.base import BaseObject


class BaseDeepNetwork(BaseObject, ABC):
    """Abstract base class for deep learning networks."""

    _tags = {
        "python_dependencies": "tensorflow",
        "python_version": "<3.11",
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
