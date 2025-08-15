"""TS2vec tests."""

import glob
import tempfile

import numpy as np
import pytest

from aeon.transformations.collection.self_supervised import TS2Vec
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if required soft dependency torch not available",
)
@pytest.mark.parametrize("expected_feature_size", [3, 5, 10])
@pytest.mark.parametrize("n_series", [1, 2, 5])
@pytest.mark.parametrize("n_channels", [1, 2, 3])
@pytest.mark.parametrize("series_length", [3, 10, 20])
def test_ts2vec_output_shapes(
    expected_feature_size, n_series, n_channels, series_length
):
    """Test the output shapes of the TS2Vec transformer."""
    X = np.random.random(size=(n_series, n_channels, series_length))
    with tempfile.TemporaryDirectory() as tmp:
        transformer = TS2Vec(
            output_dim=expected_feature_size, device="cpu", n_epochs=2, file_path=tmp
        )
        X_t = transformer.fit_transform(X)
        assert X_t.shape == (n_series, expected_feature_size)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if required soft dependency torch not available",
)
def test_ts2vec_callbacks():
    """Test that callbacks are called during training."""
    call_count = {
        "after_epoch_callback_count": 0,
        "after_iter_callback_count": 0,
    }

    def after_epoch_callback(model, loss):
        call_count["after_epoch_callback_count"] += 1

    def after_iter_callback(model, loss):
        call_count["after_iter_callback_count"] += 1

    n_epochs = 4
    with tempfile.TemporaryDirectory() as tmp:
        transformer = TS2Vec(
            n_epochs=n_epochs,
            after_epoch_callbacks=[after_epoch_callback],
            after_iter_callbacks=after_iter_callback,
            device="cpu",
            file_path=tmp,
        )
        X = np.random.random(size=(2, 2, 5))
        transformer.fit(X)
        assert call_count["after_epoch_callback_count"] == n_epochs
        assert call_count["after_iter_callback_count"] > 0


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if required soft dependency torch not available",
)
@pytest.mark.parametrize("lr", [0.01, 0.005])
def test_ts2vec_learning_rate(lr):
    """Test that the learning rate is set correctly."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:
        transformer = TS2Vec(
            n_epochs=1,
            lr=lr,
            device="cpu",
            file_path=tmp,
            output_dim=2,
        )
        X_transformed = transformer.fit_transform(X)
        assert len(X_transformed.shape) == 2
        assert int(X_transformed.shape[-1]) == 2


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if required soft dependency torch not available",
)
@pytest.mark.parametrize("batch_size", [4, 8])
def test_ts2vec_batch_size(batch_size):
    """Test TS2Vec with different batch sizes."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:
        transformer = TS2Vec(
            batch_size=batch_size,
            n_epochs=1,
            device="cpu",
            file_path=tmp,
            output_dim=2,
        )
        X_transformed = transformer.fit_transform(X)
        assert len(X_transformed.shape) == 2
        assert int(X_transformed.shape[-1]) == 2


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if required soft dependency torch not available",
)
def test_ts2vec_save_model():
    """Test that the model is saved correctly."""
    X = np.random.random((100, 2, 5))
    with tempfile.TemporaryDirectory() as tmp:
        transformer = TS2Vec(
            n_epochs=1,
            device="cpu",
            file_path=tmp,
            output_dim=2,
            save_best_model=True,
            save_last_model=True,
            save_init_model=True,
        )
        transformer.fit(X)
        files = glob.glob(f"{tmp}/*")
        assert len(files) == 3  # init, last, best model files


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if required soft dependency torch not available",
)
def test_ts2vec_saved_loss():
    """Test that the loss is saved correctly."""
    X = np.random.random((100, 2, 5))
    n_epochs = 10
    with tempfile.TemporaryDirectory() as tmp:
        transformer = TS2Vec(
            n_epochs=n_epochs,
            device="cpu",
            file_path=tmp,
            output_dim=2,
        )
        transformer.fit(X)
        assert "loss" in transformer.history
        assert len(transformer.history["loss"]) == n_epochs
