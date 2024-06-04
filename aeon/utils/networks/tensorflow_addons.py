"""Types for typing functions signatures.

The following was taken from the tensorflow_addons deprecated package.

package: https://www.tensorflow.org/addons
version: 0.23.0
file: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/types.py
"""

from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("tensorflow", severity="none"):

    import importlib
    from typing import Callable, List, Union

    import numpy as np
    import tensorflow as tf
    from tensorflow.python.keras.engine import keras_tensor

    Number = Union[
        float,
        int,
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]

    Initializer = Union[None, dict, str, Callable, tf.keras.initializers.Initializer]
    Regularizer = Union[None, dict, str, Callable, tf.keras.regularizers.Regularizer]
    Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
    Activation = Union[None, str, Callable]
    if importlib.util.find_spec("tensorflow.keras.optimizers.legacy") is not None:
        Optimizer = Union[
            tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer, str
        ]
    else:
        Optimizer = Union[tf.keras.optimizers.Optimizer, str]

    TensorLike = Union[
        List[Union[Number, list]],
        tuple,
        Number,
        np.ndarray,
        tf.Tensor,
        tf.SparseTensor,
        tf.Variable,
        keras_tensor.KerasTensor,
    ]
    FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
    AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]
