"""Base of Rocket based transformer for GPU."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["BaseROCKETGPU"]


from aeon.transformations.collection import BaseCollectionTransformer


class BaseROCKETGPU(BaseCollectionTransformer):
    """Base class for ROCKET GPU based transformers.

    Parameters
    ----------
    n_kernels : int, default = 10000
        Number of random convolutional kernels.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "algorithm_type": "convolution",
        "capability:unequal_length": False,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        n_kernels=10000,
    ):
        super().__init__()
        self.n_kernels = n_kernels

    def _get_ppv(self, x):
        import tensorflow as tf

        x_pos = tf.math.count_nonzero(tf.nn.relu(x), axis=1)
        return tf.math.divide(x_pos, x.shape[1])

    def _get_max(self, x):
        import tensorflow as tf

        return tf.math.reduce_max(x, axis=1)
