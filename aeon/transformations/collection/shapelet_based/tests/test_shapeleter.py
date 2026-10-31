"""Tests for Shapeleter transformer."""

import numpy as np

from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
    _cosine_similarity_squared_loss,
)


def test_shapeleter_transformer_features():
    """Test fit and transform output dimensions with positional embeddings."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])
    st = ShapeleterTransform(
        max_shapelets=20, ka=8.0, ko=8.0, max_iter=5, random_state=42
    )
    Xt = st.fit_transform(X, y)
    assert Xt.ndim == 2
    assert Xt.shape[0] == 5
    assert hasattr(st, "rdst_")
    assert hasattr(st, "fusion_min_")
    assert hasattr(st, "fusion_soo_")
    assert hasattr(st, "series_len_")
    assert hasattr(st, "shapelet_lengths_")


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


def test_cosine_similarity_squared_loss():
    """Test loss computation."""
    coords = np.array([[0.1, 0.2], [0.3, 0.4]])
    loss = _cosine_similarity_squared_loss(10.0, 10.0, coords)
    assert isinstance(loss, float)
    assert not np.isnan(loss)
