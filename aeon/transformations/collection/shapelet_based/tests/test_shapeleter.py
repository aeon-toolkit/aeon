"""Tests for Shapeleter transformer."""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale

from aeon.datasets import load_unit_test
from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
    _CombinationFusion,
    _cosine_similarity_squared_loss,
    _hilbert_transform,
    _hypergraph_prune,
    _knn_hyperedges,
    _optimize_position_encoding_np,
    _PairLDA,
    _series_transform,
    _vectorized_2feature_lda_fit,
    _vectorized_2feature_lda_transform,
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
    assert np.isfinite(Xt).all()
    assert set(st.ka_opt_) == set(st.unique_dilations_)
    assert set(st.ko_opt_) == set(st.unique_dilations_)


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


def test_fit_does_not_keep_training_features():
    """Test that fit() does not retain training feature groups or duplicate aliases."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])
    st = ShapeleterTransform(
        max_shapelets=20, per_shapelets=5, max_iter=2, random_state=42
    )
    st.fit(X, y)
    assert not hasattr(st, "_fit_transformed_groups")
    assert not hasattr(st, "_unique_dilations")
    assert not hasattr(st, "_base_abs")
    assert not hasattr(st, "_base_ord")


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


@pytest.mark.parametrize(
    "labels, seed, save_rate, threshold, expected",
    [
        ([0] + [1] * 5 + [2] * 5, 0, 1.0, 1.1, [9, 0, 3, 8, 4, 1, 10, 5, 7, 2, 6]),
        (
            [0] + [1] * 2 + [2] * 4 + [3] * 12,
            0,
            0.5,
            0.9,
            [13, 0, 1, 3, 9, 2, 7, 4, 6, 8],
        ),
        ([0] * 2 + [1] * 3 + [2] * 10, 3, 0.5, 0.9, [14, 1, 3, 11, 0, 4, 9, 2]),
    ],
)
def test_hypergraph_prune_matches_reference(
    labels, seed, save_rate, threshold, expected
):
    """Expected output from the reference neighbor_hypergraph_based_shapelet_selection.

    Reference: https://github.com/CCHe64/Shapeleter (Shapeleter.ipynb, cell 11),
    inputs: shapelets = RandomState(seed).normal(size=(n, 1, 11)), neighbor_k=[3, 5].
    """
    labels = np.array(labels)
    sh = np.random.RandomState(seed).normal(size=(len(labels), 1, 11))
    selected = _hypergraph_prune(
        sh, 11, labels, save_rate=save_rate, jaccard_threshold=threshold
    )
    assert selected.tolist() == expected


def _pe_inputs():
    rng = np.random.RandomState(42)
    n, k = 20, 5
    feat = rng.normal(size=(n, 2 * k))
    pos = rng.uniform(0, 30, size=(n, k))
    order = np.argsort(np.argsort(pos, axis=1), axis=1).astype(float)
    return feat, pos, order, 50.0


def test_optimize_position_encoding_regression():
    """Regression pin of this implementation (NOT yet checked against reference)."""
    feat, pos, order, len_map = _pe_inputs()
    ka, ko = _optimize_position_encoding_np(
        feat, pos, order, len_map, ka0=8.0, ko0=8.0, max_iter=20
    )
    assert ka == pytest.approx(8.325319, abs=1e-4)
    assert ko == pytest.approx(8.325612, abs=1e-4)


REF_KA = 11.41175172217355
REF_KO = 11.482662795661266


def test_optimize_position_encoding_matches_reference():
    """Verify Adam positional optimizer matches reference implementation.

    Reference: reference cells 19/20, CCHe64/Shapeleter (Shapeleter.ipynb).
    """
    feat, pos, order, len_map = _pe_inputs()
    ka, ko = _optimize_position_encoding_np(feat, pos, order, len_map)
    assert ka == pytest.approx(REF_KA, abs=1e-3)
    assert ko == pytest.approx(REF_KO, abs=1e-3)


def test_optimize_position_encoding_first_step_is_log_space():
    """Adam's first step moves log(K) by ~lr, so K changes by a factor e^(+-lr).

    A linear-space Adam would move K by ~lr (8.0 -> 8.1), not by a factor.
    """
    feat, pos, order, len_map = _pe_inputs()
    ka, ko = _optimize_position_encoding_np(
        feat, pos, order, len_map, ka0=8.0, ko0=8.0, lr=0.1, max_iter=1
    )
    for k in (ka, ko):
        assert any(k == pytest.approx(8.0 * np.exp(s * 0.1), rel=1e-3) for s in (1, -1))


def test_optimize_position_encoding_stays_in_bounds():
    """Ka and Ko stay inside [k_min, k_max] even under large learning rates."""
    feat, pos, order, len_map = _pe_inputs()
    ka, ko = _optimize_position_encoding_np(
        feat,
        pos,
        order,
        len_map,
        ka0=8.0,
        ko0=8.0,
        k_min=2.0,
        k_max=10.0,
        lr=5.0,
        max_iter=30,
    )
    assert 2.0 <= ka <= 10.0
    assert 2.0 <= ko <= 10.0


def test_optimize_position_encoding_first_step_exit_falls_back():
    """If the very first step leaves [k_min, k_max], return (ka0, ko0)."""
    feat, pos, order, len_map = _pe_inputs()
    ka, ko = _optimize_position_encoding_np(
        feat,
        pos,
        order,
        len_map,
        ka0=8.0,
        ko0=8.0,
        k_min=7.999,
        k_max=8.001,
        lr=0.5,
        max_iter=50,
    )
    assert (ka, ko) == (8.0, 8.0)


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


@pytest.mark.parametrize("mode", ["F", "H", "HF"])
def test_series_transform_matches_sklearn_scale(mode):
    """Verify _series_transform matches sklearn scale per series and handles const."""
    rng = np.random.RandomState(42)
    steps = rng.normal(size=(5, 1, 40))
    X = np.cumsum(steps, axis=-1)

    res = _series_transform(X, mode)
    if mode == "F":
        raw = np.diff(X, axis=-1)
    elif mode == "H":
        raw = _hilbert_transform(X)
    else:
        raw = np.diff(_hilbert_transform(X), axis=-1)

    expected = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        expected[i, 0] = scale(raw[i, 0])

    np.testing.assert_allclose(res, expected, atol=1e-9)

    # Constant input series for mode F (std=0 after diff)
    if mode == "F":
        X_const = np.ones((3, 1, 30))
        res_const = _series_transform(X_const, mode)
        assert np.all(np.isfinite(res_const))
        np.testing.assert_allclose(res_const, 0.0, atol=1e-12)


def _lda_pair_data(n_classes, seed=0, n=60, k=6):
    rng = np.random.RandomState(seed)
    y = np.arange(n) % n_classes
    x1 = rng.normal(size=(n, k)) + y[:, None] * rng.uniform(0, 1, k)
    x2 = 0.5 * x1 + rng.normal(size=(n, k)) + y[:, None] * rng.uniform(-1, 1, k)
    return x1, x2, y


def test_lda_projection_matches_sklearn_transform_binary():
    """Test that 2-feature LDA projection matches sklearn transform for binary y."""
    x1, x2, y = _lda_pair_data(2)
    fit = _vectorized_2feature_lda_fit(x1, x2, y)
    out = _vectorized_2feature_lda_transform(x1, x2, *fit)
    for j in range(x1.shape[1]):
        z = (
            LinearDiscriminantAnalysis(n_components=1)
            .fit(np.c_[x1[:, j], x2[:, j]], y)
            .transform(np.c_[x1[:, j], x2[:, j]])[:, 0]
        )
        np.testing.assert_allclose(out[:, j], z, atol=1e-8)


@pytest.mark.parametrize("n_classes", [3, 4])
def test_lda_projection_matches_sklearn_transform_multiclass_up_to_sign(n_classes):
    """Test that 2-feature LDA projection matches sklearn transform up to sign."""
    x1, x2, y = _lda_pair_data(n_classes)
    fit = _vectorized_2feature_lda_fit(x1, x2, y)
    out = _vectorized_2feature_lda_transform(x1, x2, *fit)
    for j in range(x1.shape[1]):
        z = (
            LinearDiscriminantAnalysis(n_components=1)
            .fit(np.c_[x1[:, j], x2[:, j]], y)
            .transform(np.c_[x1[:, j], x2[:, j]])[:, 0]
        )
        s = np.sign(out[:, j] @ z)
        np.testing.assert_allclose(s * out[:, j], z, atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shapeleter_transform_unit_test_fits(seed):
    """Regression: UnitTest used to crash on constant feature pairs."""
    X, y = load_unit_test(split="train")
    st = ShapeleterTransform(
        max_shapelets=200, per_shapelets=20, max_iter=20, random_state=seed
    )
    Xt = st.fit_transform(X, y)
    assert np.all(np.isfinite(Xt))


def test_normalise_soo_matches_loop():
    """Verify vectorized SOO normalization matches the per-shapelet reference loop."""
    st = ShapeleterTransform()
    st.series_len_ = 50
    st.shapelet_lengths_ = np.array([5, 7, 11, 15])
    st.shapelet_dilations_ = np.array([1, 2, 1, 2])
    n_shapelets = len(st.shapelet_lengths_)
    rng = np.random.RandomState(42)
    raw_feats = rng.uniform(0.0, 10.0, size=(10, 3 * n_shapelets))

    ref = raw_feats.copy()
    for i in range(n_shapelets):
        denom = (
            st.series_len_ - (st.shapelet_lengths_[i] - 1) * st.shapelet_dilations_[i]
        )
        denom = max(1.0, float(denom))
        ref[:, 2 + 3 * i] = ref[:, 2 + 3 * i] / denom

    vec = st._normalise_soo(raw_feats.copy())
    np.testing.assert_allclose(vec, ref, atol=1e-12)


@pytest.mark.parametrize(
    "invalid_param",
    [
        {"max_shapelets": 0},
        {"per_shapelets": -1},
        {"k_min": 0.0},
        {"k_min": 10.0, "k_max": 5.0},
        {"ka": 0.5, "k_min": 1.0, "k_max": 15.0},
        {"ko": 20.0, "k_min": 1.0, "k_max": 15.0},
        {"lr": 0.0},
        {"max_iter": -5},
        {"jaccard_threshold": 0.0},
        {"jaccard_threshold": 1.5},
        {"neighbor_k": ()},
        {"neighbor_k": (3, -1)},
        {"neighbor_k": (0,)},
    ],
)
def test_shapeleter_hyperparameter_validation(invalid_param):
    """Test that invalid hyperparameter values raise ValueError."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])
    st = ShapeleterTransform(**invalid_param)
    with pytest.raises(ValueError):
        st.fit(X, y)


def test_rdst_pruning_keeps_internal_state_consistent():
    """Verify pruning RDST shapelets maintains counts and transform width."""
    X = np.random.RandomState(42).normal(size=(10, 1, 40))
    y = np.array([0, 1] * 5)
    st = ShapeleterTransform(
        max_shapelets=100, per_shapelets=10, max_iter=1, random_state=42
    )
    st.fit(X, y)
    n = len(st.shapelet_lengths_)
    assert len(st.rdst_.shapelets_[0]) == n
    assert st.rdst_.transform(X).shape[1] == 3 * n


def test_rdst_feature_order_min_argmin_soo():
    """Verify RDST transform outputs [min distance, argmin position, SOO]."""
    X = np.random.RandomState(42).normal(size=(10, 1, 40))
    y = np.array([0, 1] * 5)
    st = ShapeleterTransform(
        max_shapelets=50, per_shapelets=10, max_iter=1, random_state=42
    )
    st.fit(X, y)
    feats = st.rdst_.transform(X)
    assert np.all(feats[:, 0::3] >= 0.0)
    assert np.all(np.isclose(feats[:, 1::3], np.round(feats[:, 1::3])))
    assert np.all(feats[:, 2::3] >= 0.0)


def test_shapeleter_transform_multiclass():
    """Verify multiclass y produces finite features through pairwise LDA fusion."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(12, 1, 30))
    y = np.array([0, 1, 2] * 4)
    st = ShapeleterTransform(
        max_shapelets=20, per_shapelets=5, max_iter=2, random_state=42
    )
    Xt = st.fit_transform(X, y)
    assert Xt.shape[0] == 12
    assert np.all(np.isfinite(Xt))


def test_shapeleter_neighbor_k_changes_selection():
    """Verify that changing neighbor_k alters the hypergraph-pruned shapelet set."""
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5)
    sh = np.random.RandomState(42).normal(size=(len(labels), 1, 11))
    s1 = _hypergraph_prune(sh, 11, labels, neighbor_k=(2,), save_rate=0.5)
    s2 = _hypergraph_prune(sh, 11, labels, neighbor_k=(5,), save_rate=0.5)
    assert not np.array_equal(s1, s2)


def test_shapeleter_pruning_normalization_order():
    """Verify hypergraph pruning retains normalized candidates before unnormalized."""
    X, y = load_unit_test(split="train")
    st = ShapeleterTransform(
        max_shapelets=200, per_shapelets=20, max_iter=0, random_state=0
    )
    st.fit(X, y)
    dilas = st.rdst_.shapelets_[3]
    norms = st.rdst_.shapelets_[5]
    for d in np.unique(dilas):
        d_norms = norms[dilas == d]
        if 1 in d_norms and 0 in d_norms:
            first_zero = np.where(d_norms == 0)[0][0]
            last_one = np.where(d_norms == 1)[0][-1]
            assert last_one < first_zero, "Expected norm=1 shapelets to precede norm=0"


def _ref_pair_transform_single(x1_tr, x2_tr, y_tr, x1_te, x2_te, thr=20.0):
    """Project a single column pair using scikit-learn LDA."""
    lda = LinearDiscriminantAnalysis(n_components=1)
    try:
        res = lda.fit_transform(np.c_[x1_tr, x2_tr], y_tr)
        if res.shape[1] == 1:
            out_train = np.clip(res[:, 0], None, thr)
            out_test = np.clip(lda.transform(np.c_[x1_te, x2_te])[:, 0], None, thr)
            return out_train, out_test
    except Exception:
        pass
    out_train = 0.5 * (x1_tr + x2_tr)
    out_test = 0.5 * (x1_te + x2_te)
    return out_train, out_test


def _make_crafted_columns(N, n_classes, seed=42):
    """Construct 8 characteristic feature-pair types spanning conditioning regimes."""
    rng = np.random.RandomState(seed)
    y = np.repeat(np.arange(n_classes), int(np.ceil(N / n_classes)))[:N]
    c0_1, c0_2 = rng.randn(N), rng.randn(N)
    c1_1 = rng.randn(N)
    c1_2 = c1_1 + 1e-12 * rng.randn(N)
    c2_1, c2_2 = np.full(N, -1.0), rng.randn(N)
    c3_1, c3_2 = rng.randn(N), np.full(N, -1.0)
    c4_1, c4_2 = np.full(N, -1.0), np.full(N, -1.0)
    c5_1, c5_2 = rng.randn(N), rng.randn(N)
    c5_1[y == 0] = -1.0
    c6_1, c6_2 = rng.randn(N) * 5.0, rng.randn(N) * 5.0
    c7_1, c7_2 = rng.randn(N) * 1e3, rng.randn(N) * 1e3
    X1 = np.column_stack([c0_1, c1_1, c2_1, c3_1, c4_1, c5_1, c6_1, c7_1])
    X2 = np.column_stack([c0_2, c1_2, c2_2, c3_2, c4_2, c5_2, c6_2, c7_2])
    return X1, X2, y


def _make_crafted_test(N_t, seed=142):
    """Construct test set for the 8 characteristic feature-pair types."""
    rng = np.random.RandomState(seed)
    c0_1, c0_2 = rng.randn(N_t), rng.randn(N_t)
    c1_1 = rng.randn(N_t)
    c1_2 = c1_1 + 1e-12 * rng.randn(N_t)
    c2_1, c2_2 = np.full(N_t, -1.0), rng.randn(N_t)
    c3_1, c3_2 = rng.randn(N_t), np.full(N_t, -1.0)
    c4_1, c4_2 = np.full(N_t, -1.0), np.full(N_t, -1.0)
    c5_1, c5_2 = rng.randn(N_t), rng.randn(N_t)
    c6_1, c6_2 = rng.randn(N_t) * 20.0 + 30.0, rng.randn(N_t) * 20.0 + 30.0
    c7_1, c7_2 = rng.randn(N_t) * 1e3, rng.randn(N_t) * 1e3
    X1_te = np.column_stack([c0_1, c1_1, c2_1, c3_1, c4_1, c5_1, c6_1, c7_1])
    X2_te = np.column_stack([c0_2, c1_2, c2_2, c3_2, c4_2, c5_2, c6_2, c7_2])
    return X1_te, X2_te


def test_combination_fusion_hybrid_parity_binary():
    """Verify hybrid _PairLDA matches reference scikit-learn helper on binary y."""
    X1_tr, X2_tr, y_tr = _make_crafted_columns(30, 2)
    X1_te, X2_te = _make_crafted_test(60)
    pair = _PairLDA(threshold=20.0, safe=0.1)
    pair.fit(X1_tr, X2_tr, y_tr)
    out_tr = pair.transform(X1_tr, X2_tr)
    out_te = pair.transform(X1_te, X2_te)
    for j in range(8):
        h_tr, h_te = _ref_pair_transform_single(
            X1_tr[:, j], X2_tr[:, j], y_tr, X1_te[:, j], X2_te[:, j]
        )
        np.testing.assert_allclose(out_tr[:, j], h_tr, atol=1e-6)
        np.testing.assert_allclose(out_te[:, j], h_te, atol=1e-6)


def test_combination_fusion_hybrid_parity_multiclass():
    """Verify hybrid _PairLDA matches reference helper on 3-class y."""
    X1_tr, X2_tr, y_tr = _make_crafted_columns(30, 3)
    X1_te, X2_te = _make_crafted_test(60)
    pair = _PairLDA(threshold=20.0, safe=0.1)
    pair.fit(X1_tr, X2_tr, y_tr)
    out_tr = pair.transform(X1_tr, X2_tr)
    out_te = pair.transform(X1_te, X2_te)
    for j in range(8):
        h_tr, h_te = _ref_pair_transform_single(
            X1_tr[:, j], X2_tr[:, j], y_tr, X1_te[:, j], X2_te[:, j]
        )
        if j in pair.use_closed:
            diff_tr = min(
                np.max(np.abs(out_tr[:, j] - h_tr)),
                np.max(np.abs(out_tr[:, j] + h_tr)),
            )
            diff_te = min(
                np.max(np.abs(out_te[:, j] - h_te)),
                np.max(np.abs(out_te[:, j] + h_te)),
            )
            assert diff_tr < 1e-6
            assert diff_te < 1e-6
        else:
            np.testing.assert_allclose(out_tr[:, j], h_tr, atol=1e-12)
            np.testing.assert_allclose(out_te[:, j], h_te, atol=1e-12)


def test_combination_fusion_closed_form_used():
    """Verify _CombinationFusion uses closed-form LDA on well-conditioned data."""
    rng = np.random.RandomState(42)
    N = 30
    x_orig = rng.randn(N, 10)
    x_abs = rng.randn(N, 10)
    x_ord = rng.randn(N, 10)
    y = np.array([0, 1] * 15)
    fusion = _CombinationFusion()
    fusion.fit_transform(x_orig, x_abs, x_ord, y)
    assert fusion.n_closed_ > 0
