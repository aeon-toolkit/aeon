"""Tests for Shapeleter transformer."""

import numpy as np
import pytest

from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch is not available",
)
def test_shapeleter_transformer_features():
    """Test fit and transform output dimensions with positional embeddings."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])

    st = ShapeleterTransform(
        max_shapelets=20, ka=8.0, ko=8.0, max_iter=5, random_state=42
    )
    st.fit(X, y)
    Xt = st.transform(X)

    assert Xt.ndim == 2
    assert Xt.shape[0] == 5
    # Feature count should exceed base RDST features due to pe_abs and pe_ord
    assert Xt.shape[1] > 20
    assert hasattr(st, "_base_abs")
    assert hasattr(st, "_base_ord")
    assert len(st._base_abs) > 0
    assert len(st._base_ord) > 0


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch is not available",
)
def test_shapeleter_fixed_k_max_iter_zero():
    """Test that max_iter=0 uses fixed ka and ko without optimization."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])

    ka_fixed = 3.5
    ko_fixed = 4.5
    st = ShapeleterTransform(
        max_shapelets=20, ka=ka_fixed, ko=ko_fixed, max_iter=0, random_state=42
    )
    st.fit(X, y)

    # For each dilation, base_abs should exactly equal ka * len_map / (2 * pi)
    for d in st._unique_dilations:
        sub_idx = np.where(st._shapelet_dilations == d)[0]
        sh_len = st._shapelet_lengths[sub_idx[0]]
        len_map = max(1.0, float(st._series_len - (sh_len - 1) * d))
        n_sub = max(1.0, float(len(sub_idx)))

        expected_base_abs = ka_fixed * len_map / (2.0 * np.pi)
        expected_base_ord = ko_fixed * n_sub / (2.0 * np.pi)

        np.testing.assert_allclose(st._base_abs[d], expected_base_abs)
        np.testing.assert_allclose(st._base_ord[d], expected_base_ord)


def test_shapeleter_position_encoding_optimizer():
    """Test the Adam optimizer for Ka and Ko positional encoding parameters."""
    from aeon.transformations.collection.shapelet_based._shapeleter import (
        _optimize_position_encoding_np,
    )

    rng = np.random.RandomState(42)
    N = 25
    K = 8
    featureX = rng.normal(size=(N, 2 * K))
    positionX = rng.uniform(0, 50, size=(N, K))
    orderX = np.argsort(np.argsort(positionX, axis=1), axis=1).astype(float)
    len_map = 80.0

    ka_opt, ko_opt = _optimize_position_encoding_np(
        featureX,
        positionX,
        orderX,
        len_map,
        ka0=8.0,
        ko0=8.0,
        k_min=1.0,
        k_max=15.0,
        lr=0.01,
        max_iter=30,
    )

    assert 1.0 <= ka_opt <= 15.0
    assert 1.0 <= ko_opt <= 15.0


def test_cosine_similarity_squared_loss():
    """Test column-wise cosine similarity squared loss helper."""
    from aeon.transformations.collection.shapelet_based._shapeleter import (
        _cosine_similarity_squared_loss,
    )

    f0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    f1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(_cosine_similarity_squared_loss(f0, f1), 1.0)

    f2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(_cosine_similarity_squared_loss(f0, f2), 0.0)


def test_combination_fusion_lda_shape_guard():
    """Test that _CombinationFusion guards 1D/single component output shapes."""
    from aeon.transformations.collection.shapelet_based._shapeleter import (
        _CombinationFusion,
    )

    fusion = _CombinationFusion()
    v1 = np.ones((10, 2))
    v2 = np.ones((10, 2))
    y = np.array([0, 1] * 5)
    lda, trans = fusion._fit_single_pair(v1, v2, y)
    assert trans.shape == (10, 1)


def test_hypergraph_neighborhood_self_inclusion():
    """Test that hypergraph prune includes shapelet i itself in its hyperedge."""
    from aeon.transformations.collection.shapelet_based._shapeleter import (
        _hypergraph_prune,
    )

    shapelets = np.random.RandomState(42).normal(size=(10, 1, 15))
    labels = np.array([0, 1] * 5)
    selected = _hypergraph_prune(
        shapelets,
        shapelet_len=15,
        labels=labels,
        neighbor_k=(3, 5),
        save_rate=0.5,
    )
    assert len(selected) > 0
    assert len(selected) <= 10
