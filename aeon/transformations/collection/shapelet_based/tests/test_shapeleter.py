"""Tests for Shapeleter transformer."""

import numpy as np
import pytest

from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
    _CombinationFusion,
    _cosine_similarity_squared_loss,
    _hypergraph_prune,
    _knn_hyperedges,
    _optimize_position_encoding_np,
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


def test_shapeleter_transformer_fit_transform_equivalence():
    """Test that fit_transform and fit then transform produce equivalent output."""
    X = np.random.RandomState(42).normal(size=(6, 1, 30))
    y = np.array([0, 1, 0, 1, 0, 1])
    st1 = ShapeleterTransform(
        max_shapelets=20, per_shapelets=5, max_iter=2, random_state=42
    )
    xt1 = st1.fit_transform(X, y)

    st2 = ShapeleterTransform(
        max_shapelets=20, per_shapelets=5, max_iter=2, random_state=42
    )
    st2.fit(X, y)
    xt2 = st2.transform(X)

    np.testing.assert_allclose(xt1, xt2, atol=1e-5)


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
    """Test loss computation matches exact hand-calculated value."""
    f0 = np.array([[0.1, 0.2], [0.3, 0.4]])
    f1 = np.array([[0.2, 0.1], [0.4, 0.3]])
    loss = _cosine_similarity_squared_loss(f0, f1)
    assert loss == pytest.approx(0.98, abs=1e-5)


def test_hypergraph_neighborhood_self_inclusion():
    """k-NN hyperedges must include the node itself."""
    cleaned = np.array([[0.0], [10.0], [20.0]])
    edges = _knn_hyperedges(cleaned, valid_k=[1, 2])
    for i in range(3):
        assert i in edges[i]
        assert i in edges[3 + i]


def test_hypergraph_prune_reduces_size():
    """Hypergraph pruning actually reduces the candidate count."""
    rng = np.random.RandomState(42)
    n = 20
    shapelets = rng.normal(size=(n, 1, 8))
    labels = np.array([0, 1] * (n // 2))
    pruned = _hypergraph_prune(shapelets, 8, labels, save_rate=0.25)
    assert len(pruned) < n
    assert len(pruned) == int(np.ceil(0.25 * n))


def test_hypergraph_prune_exhausted_label_rotation():
    """When a class is exhausted, rotation picks the NEXT surviving class."""
    shapelets = np.arange(11, dtype=float).reshape(11, 1, 1)
    labels = np.array([0] * 1 + [1] * 5 + [2] * 5)
    pruned = _hypergraph_prune(shapelets, 1, labels, neighbor_k=(2,), save_rate=1.0)
    pruned_labels = labels[pruned]
    assert (pruned_labels == 0).sum() == 1
    assert (pruned_labels == 1).sum() > 1
    assert (pruned_labels == 2).sum() > 1


def test_optimize_position_encoding_pinned():
    """Test that Adam positional optimization matches pinned reference values."""
    rng = np.random.RandomState(42)
    n, k = 20, 5
    feat = rng.normal(size=(n, 2 * k))
    pos = rng.uniform(0, 30, size=(n, k))
    order = np.argsort(np.argsort(pos, axis=1), axis=1).astype(float)
    len_map = 50.0

    ka, ko = _optimize_position_encoding_np(
        feat, pos, order, len_map, ka0=8.0, ko0=8.0, max_iter=20
    )
    assert ka == pytest.approx(8.325319, abs=1e-4)
    assert ko == pytest.approx(8.325612, abs=1e-4)


def test_combination_fusion_constant_columns():
    """Test that 2-feature LDA fusion handles constant columns without crashing."""
    rng = np.random.RandomState(42)
    n, k = 20, 10
    x_orig = rng.normal(size=(n, k))
    x_abs = rng.normal(size=(n, k))
    x_ord = rng.normal(size=(n, k))
    # Make columns 0 and 1 constant across all samples
    x_orig[:, 0] = 1.0
    x_abs[:, 0] = 1.0
    x_ord[:, 0] = 1.0
    x_orig[:, 1] = 0.0
    x_abs[:, 1] = 0.0
    x_ord[:, 1] = 0.0

    y = np.array([0, 1] * 10)
    fusion = _CombinationFusion(threshold=20.0)
    oa, or_feat, ar = fusion.fit_transform(x_orig, x_abs, x_ord, y)

    assert oa.shape == (n, k)
    assert or_feat.shape == (n, k)
    assert ar.shape == (n, k)
    assert not np.any(np.isnan(oa))
    assert not np.any(np.isnan(or_feat))
    assert not np.any(np.isnan(ar))

    oa_test, or_test, ar_test = fusion.transform(x_orig, x_abs, x_ord)
    np.testing.assert_allclose(oa, oa_test, atol=1e-5)
    np.testing.assert_allclose(or_feat, or_test, atol=1e-5)
    np.testing.assert_allclose(ar, ar_test, atol=1e-5)


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
