"""Abstract base class for deep learning networks."""

__author__ = ["hadifawaz1999", "Withington", "TonyBagnall"]

from abc import ABC, abstractmethod

from aeon.base import BaseObject
from aeon.utils.validation._dependencies import _check_soft_dependencies


class BaseDeepNetwork(BaseObject, ABC):
    """Abstract base class for deep learning networks."""

    def plot_network(
        self,
        input_shape,
        file_name=None,
        show_shapes=True,
        show_layer_names=False,
        show_layer_activations=True,
        dpi=96,
    ):
        """Plot the network with its input-output shapes and activations.

        Parameters
        ----------
        input_shape : tuple
                      The shape of the data fed into the input layer.
        file_name : str default = None ("model.pdf")
                    File name of the plot image, without the extension.
        show_shapes : bool, default = True
                      Whether to display shape information.
        show_layer_names : bool, default = False
                           Whether to display layer names.
        show_layer_activations : bool, default = True
                                 Display layer activations
                                 (only for layers that have an activation property).
        dpi : int, default = 96
              Dots per inch.

        Returns
        -------
        None
        """
        _check_soft_dependencies("tensorflow")
        import tensorflow as tf

        input_layer, output_layer = self.build_network(input_shape=input_shape)
        network = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        file_name_ = "model" if file_name is None else file_name

        try:
            tf.keras.utils.plot_model(
                model=network,
                to_file=file_name_ + ".pdf",
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                show_layer_activations=show_layer_activations,
                dpi=dpi,
            )
        except ImportError:
            raise ImportError("Either graphviz or pydot should be installed.")

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
