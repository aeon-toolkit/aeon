"""Tests for the Weight Normalization layer."""

import os

import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="soft dependency tensorflow not found in the system",
)
def test_weight_norm():
    """Test the weight norm layer."""
    import numpy as np
    import tensorflow as tf

    from aeon.utils.networks.weight_norm import _WeightNormalization

    X = np.random.random((10, 10, 5))
    _input = tf.keras.layers.Input((10, 5))
    l1 = _WeightNormalization(
        tf.keras.layers.Conv1D(filters=5, kernel_size=1, dilation_rate=4)
    )(_input)
    model = tf.keras.models.Model(inputs=_input, outputs=l1)
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    )
    assert model is not None
    output = model.predict(X)

    assert output.shape == (
        10,
        10,
        5,
    ), f"Expected output shape (10, 10, 5), but got {output.shape}"
    assert model.layers[1].weights is not None
    assert len(model.layers[1].weights) == 4

    model_path = "test_weight_norm_model.h5"
    model.save(model_path)
    loaded_model = tf.keras.models.load_model(
        model_path, custom_objects={"_WeightNormalization": _WeightNormalization}
    )
    assert loaded_model is not None
    loaded_output = loaded_model.predict(X)
    np.testing.assert_allclose(
        output,
        loaded_output,
        err_msg="Loaded model's output differs from original model's output",
    )
    if os.path.exists(model_path):
        os.remove(model_path)
