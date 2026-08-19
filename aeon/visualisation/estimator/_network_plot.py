__all__ = ["plot_network"]
__maintainer__ = []

from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_network(
    input_shape,
    network,
    file_name=None,
    show_shapes=True,
    show_layer_names=False,
    show_layer_activations=True,
    dpi=96,
):
    """Plot the network with its input-output shapes and activations.

    The network is plotted regardless of the output task.
    Its important to note that to be able to use this function,
    the pydot package should be installed in the environment as well
    as graphviz on the system.
    The function also works when the network is an auto-encoder based
    model, and the function will split the output files into 2 files,
    the first for the encoder and the second for the decoder.

    Parameters
    ----------
    input_shape : tuple
                  The shape of the data fed into the input layer.
    network : an element from the network module
              Example: FCNNetwork().
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
    _check_soft_dependencies(["tensorflow", "pydot"])
    import tensorflow as tf

    input_layer, output_layer = network.build_network(input_shape=input_shape)

    if not isinstance(input_layer, tf.keras.models.Model):
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

    else:
        file_name_ = "model" if file_name is None else file_name

        try:
            tf.keras.utils.plot_model(
                model=input_layer,
                to_file=file_name_ + "_encoder.pdf",
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                show_layer_activations=show_layer_activations,
                dpi=dpi,
            )
            tf.keras.utils.plot_model(
                model=output_layer,
                to_file=file_name_ + "_decoder.pdf",
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                show_layer_activations=show_layer_activations,
                dpi=dpi,
            )
        except ImportError:
            raise ImportError("Either graphviz or pydot should be installed.")
