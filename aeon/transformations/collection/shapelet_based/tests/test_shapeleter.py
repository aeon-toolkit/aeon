"""Tests for Shapeleter transformer."""

import numpy as np

from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
    _cosine_similarity_squared_loss,
    _series_transform,
)


def test_shapeleter_transformer_features():
    """Test fit and transform output dimensions with positional embeddings."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])
    st = ShapeleterTransform(
        max_shapelets=20,
        per_shapelets=5,
        ka=8.0,
        ko=8.0,
        max_iter=5,
        random_state=42,
    )
    Xt = st.fit_transform(X, y)
    assert Xt.ndim == 2
    assert Xt.shape[0] == 5
    assert hasattr(st, "rdst_")
    assert hasattr(st, "fusion_min_")
    assert hasattr(st, "fusion_soo_")
    assert hasattr(st, "series_len_")
    assert hasattr(st, "shapelet_lengths_")
    assert hasattr(st, "scaler_min_")
    assert hasattr(st, "scaler_soo_")
    assert hasattr(st, "ka_opt_")
    assert hasattr(st, "ko_opt_")


def test_shapeleter_fixed_k_max_iter_zero():
    """Test initialization when max_iter is 0."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])
    st = ShapeleterTransform(
        max_shapelets=10, ka=5.0, ko=5.0, max_iter=0, random_state=42
    )
    Xt = st.fit_transform(X, y)
    assert Xt.shape[0] == 5
    assert hasattr(st, "rdst_")
    assert hasattr(st, "series_len_")
    assert hasattr(st, "ka_opt_")
    assert hasattr(st, "ko_opt_")


def test_cosine_similarity_squared_loss():
    """Test loss computation on 2D arrays."""
    f0 = np.array([[0.1, 0.2], [0.3, 0.4]])
    f1 = np.array([[0.2, 0.1], [0.4, 0.3]])
    loss = _cosine_similarity_squared_loss(f0, f1)
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert 0.0 <= loss <= 1.0


def test_series_transform_modes():
    """Test F, H, and HF series transformations."""
    X = np.random.RandomState(42).normal(size=(4, 1, 30))
    X_F = _series_transform(X, "F")
    X_H = _series_transform(X, "H")
    X_HF = _series_transform(X, "HF")

    assert X_F.shape == (4, 1, 29)
    assert X_H.shape == (4, 1, 30)
    assert X_HF.shape == (4, 1, 29)

    np.testing.assert_allclose(np.mean(X_F, axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(X_F, axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.mean(X_H, axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(X_H, axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.mean(X_HF, axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(X_HF, axis=-1), 1.0, atol=1e-5)
