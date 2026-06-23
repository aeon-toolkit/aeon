"""Tests whether various clusterer params work well."""

import os
import tempfile

import numpy as np
import pytest

from aeon.clustering.deep_learning import (
    AEAttentionBiGRUClusterer,
    AEBiGRUClusterer,
    AEDCNNClusterer,
    AEDRNNClusterer,
    AEFCNClusterer,
    AEResNetClusterer,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies

ALL_DEEP_CLUSTERERS = [
    AEAttentionBiGRUClusterer,
    AEBiGRUClusterer,
    AEDCNNClusterer,
    AEDRNNClusterer,
    AEFCNClusterer,
    AEResNetClusterer,
]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
def test_multi_rec_fcn():
    """Tests whether multi-rec loss works fine or not."""
    X = np.random.random((100, 5, 2))
    clst = AEFCNClusterer(**AEFCNClusterer._get_test_params()[0], loss="multi_rec")
    clst.fit(X)
    assert isinstance(clst.history["loss"][-1], float)

    clst = AEResNetClusterer(
        **AEResNetClusterer._get_test_params()[0], loss="multi_rec"
    )
    clst.fit(X)
    assert isinstance(clst.history["loss"][-1], float)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
def test_validation_split():
    """Test validation_split parameter."""
    X = np.random.random((100, 5, 2))

    models = [
        AEFCNClusterer,
        AEResNetClusterer,
        AEDCNNClusterer,
        AEDRNNClusterer,
        AEAttentionBiGRUClusterer,
        AEBiGRUClusterer,
    ]

    for model in models:
        # Test with validation_split = 0 (no validation)
        clst = model(
            **model._get_test_params()[0],
            validation_split=0,
        )
        clst.fit(X)
        assert "val_loss" not in clst.history.history or all(
            v is None for v in clst.history.history["val_loss"]
        )

        # Test with validation_split = 0.2 (20% validation)
        clst = model(
            **model._get_test_params()[0],
            validation_split=0.2,
        )
        clst.fit(X)
        assert "val_loss" in clst.history.history
        assert len(clst.history.history["val_loss"]) == len(
            clst.history.history["loss"]
        )
        assert all(isinstance(v, float) for v in clst.history.history["val_loss"])


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
@pytest.mark.parametrize("model", ALL_DEEP_CLUSTERERS)
def test_deep_clusterer_fit_options(model):
    """Test optional fit flags: metrics, model saving, mini-batch and callbacks."""
    import tensorflow as tf

    # use >= 20 cases so use_mini_batch_size yields a batch size of at least 2
    X = np.random.random((20, 1, 12))
    params = model._get_test_params()[0]
    with tempfile.TemporaryDirectory() as tmp:
        file_path = tmp + "/"
        clst = model(
            **params,
            metrics="mean_squared_error",
            save_init_model=True,
            use_mini_batch_size=True,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss")],
            save_last_model=True,
            file_path=file_path,
        )
        clst.fit(X)
        assert clst._metrics == ["mean_squared_error"]
        # save_init_model and save_last_model should have written .keras files
        assert os.path.exists(file_path + clst.init_file_name + ".keras")
        assert os.path.exists(file_path + clst.last_file_name + ".keras")
        assert clst.predict(X).shape == (20,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
@pytest.mark.parametrize("model", [AEFCNClusterer, AEResNetClusterer])
def test_multi_rec_validation_split(model):
    """Test the multi_rec loss path together with a validation split."""
    # use the same shape as the other multi_rec tests; a smaller collection
    # leaves the autoencoder's latent space near-constant after a couple of
    # epochs, which the downstream clusterer's variation check rejects.
    X = np.random.random((100, 5, 2))
    clst = model(
        **model._get_test_params()[0],
        loss="multi_rec",
        validation_split=0.2,
    )
    clst.fit(X)
    assert "val_loss" in clst.history
    assert len(clst.history["val_loss"]) == len(clst.history["loss"])


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
def test_get_model_checkpoint_callback_single():
    """Test the checkpoint callback helper wraps a single (non-list) callback."""
    import tensorflow as tf

    clst = AEFCNClusterer(**AEFCNClusterer._get_test_params()[0])
    single = tf.keras.callbacks.EarlyStopping(monitor="loss")
    out = clst._get_model_checkpoint_callback(
        callbacks=single, file_path="./", file_name="unused"
    )
    assert len(out) == 2
    assert out[0] is single
