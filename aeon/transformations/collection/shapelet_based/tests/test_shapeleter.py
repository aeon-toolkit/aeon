"""Tests for Shapeleter transformer."""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler, scale

from aeon.datasets import load_unit_test
from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransform,
    _CombinationFusion,
    _cosine_similarity_squared_loss,
    _hilbert_transform,
    _hypergraph_prune,
    _jaccard_similar_neighbors,
    _knn_hyperedges,
    _optimize_position_encoding_np,
    _series_transform,
    _sk_lda_divides_by_n_minus_classes,
    _sk_lda_pair_fit,
)


def test_shapeleter_transformer_features():
    """Test fit and transform output dimensions with positional embeddings."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])
    st = ShapeleterTransform(
        max_shapelets=10,
        per_shapelets=3,
        ka=8.0,
        ko=8.0,
        max_iter=1,
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


def _build_test_columns(n_classes, seed=0):
    """Build train and test sets spanning 10 characteristic conditioning regimes."""
    rng = np.random.RandomState(seed)
    n = 36
    y = rng.permutation(np.arange(n) % n_classes)
    y_te = np.arange(60) % n_classes

    cols_tr_x1, cols_tr_x2 = [], []
    cols_te_x1, cols_te_x2 = [], []

    # 1: normal
    cols_tr_x1.append(rng.randn(n) + 0.5 * y)
    cols_tr_x2.append(rng.randn(n) + 0.5 * y)
    cols_te_x1.append(rng.randn(60) + 0.5 * y_te)
    cols_te_x2.append(rng.randn(60) + 0.5 * y_te)

    # 2: x1 constant -1.0 with x2 small variance class dependent
    cols_tr_x1.append(np.full(n, -1.0))
    cols_tr_x2.append(0.01 * rng.randn(n) + 0.02 * y)
    cols_te_x1.append(np.full(60, -1.0))
    cols_te_x2.append(0.01 * rng.randn(60) + 0.02 * y_te)

    # 3: x1 constant with x2 class independent small variance
    cols_tr_x1.append(np.full(n, -1.0))
    cols_tr_x2.append(0.01 * rng.randn(n))
    cols_te_x1.append(np.full(60, -1.0))
    cols_te_x2.append(0.01 * rng.randn(60))

    # 4: x2 constant -1.0
    cols_tr_x1.append(rng.randn(n) + 0.5 * y)
    cols_tr_x2.append(np.full(n, -1.0))
    cols_te_x1.append(rng.randn(60) + 0.5 * y_te)
    cols_te_x2.append(np.full(60, -1.0))

    # 5: both constant
    cols_tr_x1.append(np.full(n, -1.0))
    cols_tr_x2.append(np.full(n, -1.0))
    cols_te_x1.append(np.full(60, -1.0))
    cols_te_x2.append(np.full(60, -1.0))

    # 6: collinear x2 = 2 * x1 + 1
    x1_6 = rng.randn(n) + 0.5 * y
    cols_tr_x1.append(x1_6)
    cols_tr_x2.append(2.0 * x1_6 + 1.0)
    x1_6_te = rng.randn(60) + 0.5 * y_te
    cols_te_x1.append(x1_6_te)
    cols_te_x2.append(2.0 * x1_6_te + 1.0)

    # 7: collinear x2 = -3 * x1 + 2
    x1_7 = rng.randn(n) + 0.5 * y
    cols_tr_x1.append(x1_7)
    cols_tr_x2.append(-3.0 * x1_7 + 2.0)
    x1_7_te = rng.randn(60) + 0.5 * y_te
    cols_te_x1.append(x1_7_te)
    cols_te_x2.append(-3.0 * x1_7_te + 2.0)

    # 8: near collinear x2 = x1 + 0.001 * N(0,1)
    x1_8 = rng.randn(n) + 0.5 * y
    cols_tr_x1.append(x1_8)
    cols_tr_x2.append(x1_8 + 0.001 * rng.randn(n))
    x1_8_te = rng.randn(60) + 0.5 * y_te
    cols_te_x1.append(x1_8_te)
    cols_te_x2.append(x1_8_te + 0.001 * rng.randn(60))

    # 9: tiny variance
    cols_tr_x1.append(0.001 * rng.randn(n) + 0.003 * y)
    cols_tr_x2.append(0.001 * rng.randn(n) + 0.003 * y)
    cols_te_x1.append(0.001 * rng.randn(60) + 0.003 * y_te)
    cols_te_x2.append(0.001 * rng.randn(60) + 0.003 * y_te)

    # 10: well separated
    cols_tr_x1.append(rng.randn(n) + 3.0 * y)
    cols_tr_x2.append(rng.randn(n) + 3.0 * y)
    cols_te_x1.append(rng.randn(60) + 3.0 * y_te)
    cols_te_x2.append(rng.randn(60) + 3.0 * y_te)

    x1_tr = np.column_stack(cols_tr_x1)
    x2_tr = np.column_stack(cols_tr_x2)
    x1_te = np.column_stack(cols_te_x1)
    x2_te = np.column_stack(cols_te_x2)
    return x1_tr, x2_tr, y, x1_te, x2_te, y_te


@pytest.mark.parametrize("n_classes", [2, 3, 4, 6])
def test_sk_lda_pair_matches_sklearn(n_classes):
    """Verify _sk_lda_pair_fit exactly matches scikit-learn LDA SVD solver."""
    x1_tr, x2_tr, y_tr, x1_te, x2_te, y_te = _build_test_columns(n_classes)
    params = _sk_lda_pair_fit(x1_tr, x2_tr, y_tr)
    k = x1_tr.shape[1]
    failed = []
    for j in range(k):
        lda = LinearDiscriminantAnalysis(n_components=1)
        try:
            res = lda.fit_transform(np.c_[x1_tr[:, j], x2_tr[:, j]], y_tr)
            if res.shape[1] != 1:
                failed.append(j)
            else:
                proj_tr = (x1_tr[:, j] - params["xbar1"][j]) * params["s1"][j] + (
                    x2_tr[:, j] - params["xbar2"][j]
                ) * params["s2"][j]
                proj_te = (x1_te[:, j] - params["xbar1"][j]) * params["s1"][j] + (
                    x2_te[:, j] - params["xbar2"][j]
                ) * params["s2"][j]
                exp_tr = res[:, 0]
                exp_te = lda.transform(np.c_[x1_te[:, j], x2_te[:, j]])[:, 0]
                np.testing.assert_allclose(proj_tr, exp_tr, atol=1e-8)
                np.testing.assert_allclose(proj_te, exp_te, atol=1e-8)
        except (ValueError, np.linalg.LinAlgError, IndexError):
            failed.append(j)

    ok_false = [j for j in range(k) if not params["ok"][j]]
    assert set(failed) == set(ok_false)


def test_sk_lda_pair_degenerate_inputs():
    """Verify _sk_lda_pair_fit handles all-equal labels and n_cases <= n_classes."""
    rng = np.random.RandomState(42)
    p_a = _sk_lda_pair_fit(rng.randn(30, 5), rng.randn(30, 5), np.zeros(30))
    assert not np.any(p_a["ok"])

    p_b = _sk_lda_pair_fit(rng.randn(3, 5), rng.randn(3, 5), np.array([0, 1, 2]))
    assert not np.any(p_b["ok"])


def _ref_pair_transform_single(x1_tr, x2_tr, y_tr, x1_te, x2_te, thr=20.0):
    """Project a pair of column sets using scikit-learn LDA with reference clipping."""
    lda = LinearDiscriminantAnalysis(n_components=1)
    k = x1_tr.shape[1]
    out_tr = np.zeros_like(x1_tr)
    out_te = np.zeros_like(x1_te)
    for j in range(k):
        try:
            res = lda.fit_transform(np.c_[x1_tr[:, j], x2_tr[:, j]], y_tr)
            if res.shape[1] == 1:
                col_tr = res[:, 0]
                col_te = lda.transform(np.c_[x1_te[:, j], x2_te[:, j]])[:, 0]
            else:
                col_tr = 0.5 * (x1_tr[:, j] + x2_tr[:, j])
                col_te = 0.5 * (x1_te[:, j] + x2_te[:, j])
        except (ValueError, np.linalg.LinAlgError, IndexError):
            col_tr = 0.5 * (x1_tr[:, j] + x2_tr[:, j])
            col_te = 0.5 * (x1_te[:, j] + x2_te[:, j])
        col_tr = np.clip(col_tr, None, thr)
        col_te = np.clip(col_te, None, thr)
        out_tr[:, j] = col_tr
        out_te[:, j] = col_te
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(out_tr), scaler.transform(out_te)


def _build_3sets_columns(n_classes, seed=0):
    """Build three sets of 10 characteristic columns for CombinationFusion testing."""
    rng = np.random.RandomState(seed)
    n = 36
    y = rng.permutation(np.arange(n) % n_classes)
    y_te = np.arange(60) % n_classes

    cols1, cols2, cols3 = [], [], []
    cols1_te, cols2_te, cols3_te = [], [], []
    for cols, cols_te in [(cols1, cols1_te), (cols2, cols2_te), (cols3, cols3_te)]:
        cols.append(rng.randn(n) + 0.5 * y)
        cols_te.append(rng.randn(60) + 0.5 * y_te)
    cols1.append(np.full(n, -1.0))
    cols1_te.append(np.full(60, -1.0))
    cols2.append(0.01 * rng.randn(n) + 0.02 * y)
    cols2_te.append(0.01 * rng.randn(60) + 0.02 * y_te)
    cols3.append(0.01 * rng.randn(n) + 0.02 * y)
    cols3_te.append(0.01 * rng.randn(60) + 0.02 * y_te)
    cols1.append(np.full(n, -1.0))
    cols1_te.append(np.full(60, -1.0))
    cols2.append(0.01 * rng.randn(n))
    cols2_te.append(0.01 * rng.randn(60))
    cols3.append(0.01 * rng.randn(n))
    cols3_te.append(0.01 * rng.randn(60))
    cols1.append(rng.randn(n) + 0.5 * y)
    cols1_te.append(rng.randn(60) + 0.5 * y_te)
    cols2.append(np.full(n, -1.0))
    cols2_te.append(np.full(60, -1.0))
    cols3.append(rng.randn(n) + 0.5 * y)
    cols3_te.append(rng.randn(60) + 0.5 * y_te)
    cols1.append(np.full(n, -1.0))
    cols1_te.append(np.full(60, -1.0))
    cols2.append(np.full(n, -1.0))
    cols2_te.append(np.full(60, -1.0))
    cols3.append(np.full(n, -1.0))
    cols3_te.append(np.full(60, -1.0))
    x1_6 = rng.randn(n) + 0.5 * y
    x1_6_te = rng.randn(60) + 0.5 * y_te
    cols1.append(x1_6)
    cols1_te.append(x1_6_te)
    cols2.append(2.0 * x1_6 + 1.0)
    cols2_te.append(2.0 * x1_6_te + 1.0)
    cols3.append(2.0 * x1_6 + 1.0)
    cols3_te.append(2.0 * x1_6_te + 1.0)
    x1_7 = rng.randn(n) + 0.5 * y
    x1_7_te = rng.randn(60) + 0.5 * y_te
    cols1.append(x1_7)
    cols1_te.append(x1_7_te)
    cols2.append(-3.0 * x1_7 + 2.0)
    cols2_te.append(-3.0 * x1_7_te + 2.0)
    cols3.append(-3.0 * x1_7 + 2.0)
    cols3_te.append(-3.0 * x1_7_te + 2.0)
    x1_8 = rng.randn(n) + 0.5 * y
    x1_8_te = rng.randn(60) + 0.5 * y_te
    cols1.append(x1_8)
    cols1_te.append(x1_8_te)
    cols2.append(x1_8 + 0.001 * rng.randn(n))
    cols2_te.append(x1_8_te + 0.001 * rng.randn(60))
    cols3.append(x1_8 + 0.001 * rng.randn(n))
    cols3_te.append(x1_8_te + 0.001 * rng.randn(60))
    cols1.append(0.001 * rng.randn(n) + 0.003 * y)
    cols1_te.append(0.001 * rng.randn(60) + 0.003 * y_te)
    cols2.append(0.001 * rng.randn(n) + 0.003 * y)
    cols2_te.append(0.001 * rng.randn(60) + 0.003 * y_te)
    cols3.append(0.001 * rng.randn(n) + 0.003 * y)
    cols3_te.append(0.001 * rng.randn(60) + 0.003 * y_te)
    cols1.append(rng.randn(n) + 3.0 * y)
    cols1_te.append(rng.randn(60) + 3.0 * y_te)
    cols2.append(rng.randn(n) + 3.0 * y)
    cols2_te.append(rng.randn(60) + 3.0 * y_te)
    cols3.append(rng.randn(n) + 3.0 * y)
    cols3_te.append(rng.randn(60) + 3.0 * y_te)

    xo = np.column_stack(cols1)
    xa = np.column_stack(cols2)
    xr = np.column_stack(cols3)
    xo_te = np.column_stack(cols1_te)
    xa_te = np.column_stack(cols2_te)
    xr_te = np.column_stack(cols3_te)
    return xo, xa, xr, y, xo_te, xa_te, xr_te


def test_combination_fusion_matches_reference_binary():
    """Verify _CombinationFusion matches reference scikit-learn LDA for binary y."""
    xo, xa, xr, y, xo_te, xa_te, xr_te = _build_3sets_columns(2)
    fusion = _CombinationFusion()
    out_oa, out_or, out_ar = fusion.fit_transform(xo, xa, xr, y)
    te_oa, te_or, te_ar = fusion.transform(xo_te, xa_te, xr_te)

    ref_oa_tr, ref_oa_te = _ref_pair_transform_single(xo, xa, y, xo_te, xa_te)
    ref_or_tr, ref_or_te = _ref_pair_transform_single(xo, xr, y, xo_te, xr_te)
    ref_ar_tr, ref_ar_te = _ref_pair_transform_single(xa, xr, y, xa_te, xr_te)

    np.testing.assert_allclose(out_oa, ref_oa_tr, atol=1e-6)
    np.testing.assert_allclose(te_oa, ref_oa_te, atol=1e-6)
    np.testing.assert_allclose(out_or, ref_or_tr, atol=1e-6)
    np.testing.assert_allclose(te_or, ref_or_te, atol=1e-6)
    np.testing.assert_allclose(out_ar, ref_ar_tr, atol=1e-6)
    np.testing.assert_allclose(te_ar, ref_ar_te, atol=1e-6)


def test_combination_fusion_matches_reference_multiclass():
    """Verify _CombinationFusion matches reference scikit-learn LDA for 3 classes."""
    xo, xa, xr, y, xo_te, xa_te, xr_te = _build_3sets_columns(3)
    fusion = _CombinationFusion()
    out_oa, out_or, out_ar = fusion.fit_transform(xo, xa, xr, y)
    te_oa, te_or, te_ar = fusion.transform(xo_te, xa_te, xr_te)

    ref_oa_tr, ref_oa_te = _ref_pair_transform_single(xo, xa, y, xo_te, xa_te)
    ref_or_tr, ref_or_te = _ref_pair_transform_single(xo, xr, y, xo_te, xr_te)
    ref_ar_tr, ref_ar_te = _ref_pair_transform_single(xa, xr, y, xa_te, xr_te)

    np.testing.assert_allclose(out_oa, ref_oa_tr, atol=1e-6)
    np.testing.assert_allclose(te_oa, ref_oa_te, atol=1e-6)
    np.testing.assert_allclose(out_or, ref_or_tr, atol=1e-6)
    np.testing.assert_allclose(te_or, ref_or_te, atol=1e-6)
    np.testing.assert_allclose(out_ar, ref_ar_tr, atol=1e-6)
    np.testing.assert_allclose(te_ar, ref_ar_te, atol=1e-6)


def test_sk_lda_calibration_matches_installed_sklearn():
    """Verify _sk_lda_divides_by_n_minus_classes detects the installed convention."""
    _sk_lda_divides_by_n_minus_classes.cache_clear()
    rng = np.random.RandomState(0)
    y = np.arange(12) % 2
    x = rng.normal(size=(12, 2))
    x[:, 0] += y
    ref = LinearDiscriminantAnalysis(n_components=1).fit(x, y).scalings_[:, 0]
    new = _sk_lda_pair_fit(x[:, :1], x[:, 1:], y, _within_dof_old=False)
    old = _sk_lda_pair_fit(x[:, :1], x[:, 1:], y, _within_dof_old=True)
    d_new = abs(new["s1"][0] - ref[0]) + abs(new["s2"][0] - ref[1])
    d_old = abs(old["s1"][0] - ref[0]) + abs(old["s2"][0] - ref[1])

    matches_new = d_new < 1e-8
    matches_old = d_old < 1e-8
    assert matches_new ^ matches_old
    detected = _sk_lda_divides_by_n_minus_classes()
    assert detected == matches_old


def test_sk_lda_pair_equal_class_means_follows_sklearn():
    """Regression test for equal class means where LDA direction is noise-driven."""
    n_samples = 20
    y = np.arange(n_samples) % 2
    rng_base = np.random.RandomState(42)
    u = rng_base.randn(n_samples)
    x1_base = u + 0.5 * rng_base.randn(n_samples)
    x2_base = 2.0 * u + 0.5 * rng_base.randn(n_samples)
    for c in [0, 1]:
        mask = y == c
        x1_base[mask] -= x1_base[mask].mean()
        x2_base[mask] -= x2_base[mask].mean()
    x1_base += 1.0
    x2_base += 2.0

    for seed in range(20):
        rng = np.random.RandomState(seed)
        noise1 = 1e-10 * rng.randn(n_samples)
        noise2 = 1e-10 * rng.randn(n_samples)
        x1 = x1_base + noise1
        x2 = x2_base + noise2
        X = np.c_[x1, x2]

        lda = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
        ref_proj = lda.transform(X)[:, 0]

        fit = _sk_lda_pair_fit(x1[:, None], x2[:, None], y)
        assert fit["ok"][0]
        our_proj = (x1 - fit["xbar1"][0]) * fit["s1"][0] + (x2 - fit["xbar2"][0]) * fit[
            "s2"
        ][0]

        np.testing.assert_allclose(our_proj, ref_proj, atol=1e-8)


@pytest.mark.parametrize(
    "n, n_classes, seed, duplicates, save_rate, threshold",
    [
        (150, 3, 1, True, 0.125, 0.9),
        (200, 5, 2, True, 0.25, 0.5),
        (120, 2, 3, False, 0.5, 0.9),
    ],
)
def test_hypergraph_prune_larger_groups_is_valid_and_repeatable(
    n, n_classes, seed, duplicates, save_rate, threshold
):
    """Pruning on larger groups with duplicates or ties must be valid and repeatable."""
    sh = np.random.RandomState(seed).normal(size=(n, 1, 11))
    if duplicates:
        half = n // 2
        sh[half : 2 * half] = sh[:half]
    labels = np.random.RandomState(seed + 100).randint(0, n_classes, size=n)
    first = _hypergraph_prune(
        sh, 11, labels, save_rate=save_rate, jaccard_threshold=threshold
    )
    second = _hypergraph_prune(
        sh, 11, labels, save_rate=save_rate, jaccard_threshold=threshold
    )
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert first.min() >= 0 and first.max() < n
    assert 1 <= len(first) <= int(np.ceil(save_rate * n))


@pytest.mark.parametrize("save_rate", [0.125, 0.5])
def test_hypergraph_prune_duplicate_candidates_is_valid_and_repeatable(save_rate):
    """Duplicated candidates create exact ties.

    The selection must be a valid set and repeatable.
    """
    n = 150
    sh = np.random.RandomState(1).normal(size=(n, 1, 11))
    sh[n // 2 : 2 * (n // 2)] = sh[: n // 2]
    labels = np.random.RandomState(101).randint(0, 3, size=n)
    first = _hypergraph_prune(sh, 11, labels, save_rate=save_rate)
    second = _hypergraph_prune(sh, 11, labels, save_rate=save_rate)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert first.min() >= 0 and first.max() < n
    assert 1 <= len(first) <= int(np.ceil(save_rate * n))


def _ref_hypergraph_prune_plain_argsort(
    shapelets, labels, neighbor_k=(3, 5), save_rate=0.5, jaccard_threshold=0.9
):
    """Prune shapelets using reference implementation from Shapeleter.ipynb."""
    from scipy.sparse import lil_matrix
    from sklearn.metrics.pairwise import euclidean_distances

    n_shapelets = len(labels)
    dist_mat = euclidean_distances(shapelets[:, 0, :])
    np.fill_diagonal(dist_mat, -np.inf)

    edges = []
    for k in neighbor_k:
        for i in range(n_shapelets):
            edges.append(np.argsort(dist_mat[i])[:k])

    purity = np.zeros(n_shapelets)
    degree = np.zeros(n_shapelets)
    for edge in edges:
        sub = labels[edge]
        for j in range(len(edge)):
            degree[edge[j]] += 1
            purity[edge[j]] += np.sum(sub[j] == sub) / len(edge)
    purity = purity / degree

    h_mat = lil_matrix((n_shapelets, len(edges)), dtype=int)
    for edge_id, edge in enumerate(edges):
        for node in edge:
            h_mat[node, edge_id] = 1
    h_mat = h_mat.tocsr()
    inter = h_mat.dot(h_mat.T).toarray()
    degs = np.array(h_mat.sum(axis=1)).flatten()
    union = degs[:, None] + degs[None, :] - inter
    j_mat = np.zeros((n_shapelets, n_shapelets))
    np.divide(inter, union, where=(union != 0), out=j_mat)
    j_mat[union == 0] = 0.0
    np.fill_diagonal(j_mat, 1.0)

    j_bin = np.where(j_mat >= jaccard_threshold, 1.0, 0.0)
    np.fill_diagonal(j_bin, -1.0)

    unique_labels = list(np.unique(labels))
    save_n = int(np.ceil(save_rate * n_shapelets))
    top = int(np.argmax(purity))
    saved = [top]
    lbl_idx = (unique_labels.index(labels[top]) + 1) % len(unique_labels)
    sorted_nodes = np.argsort(purity)[::-1]
    candidates = sorted_nodes[sorted_nodes != top]

    for _ in range(save_n - 1):
        if len(candidates) == 0:
            break
        last = saved[-1]
        sim = np.where(j_bin[last] == 1.0)[0]
        j_bin[:, sim] = -1.0
        j_bin[sim, :] = -1.0
        j_bin[last, :] = -1.0
        j_bin[:, last] = -1.0
        candidates = candidates[~np.isin(candidates, sim)]
        if len(candidates) == 0:
            break
        cand_labels = labels[candidates]
        curr_lbl = unique_labels[lbl_idx]
        if np.sum(cand_labels == curr_lbl) == 0:
            curr_lbl_0 = curr_lbl
            lbl_idx = (lbl_idx + 1) % len(unique_labels)
            curr_lbl = unique_labels[lbl_idx]
            unique_labels = [lbl for lbl in unique_labels if lbl != curr_lbl_0]
        sub_cands = candidates[labels[candidates] == curr_lbl]
        if len(sub_cands) == 0:
            break
        top_node = sub_cands[np.argmax(purity[sub_cands])]
        lbl_idx = (lbl_idx + 1) % len(unique_labels)
        saved.append(top_node)
        candidates = candidates[candidates != top_node]

    return np.array(saved)


def test_hypergraph_prune_matches_plain_argsort_reference_tie_free():
    """Compare pruning with the plain-argsort reference on tie-free inputs."""
    from sklearn.metrics.pairwise import euclidean_distances

    n = 25
    rng = np.random.RandomState(9)
    sh = rng.normal(size=(n, 1, 11))
    labels = rng.randint(0, 2, size=n)

    # Assert pairwise distances are all distinct
    dist_mat = euclidean_distances(sh[:, 0, :])
    triu_idx = np.triu_indices(n, k=1)
    assert len(np.unique(dist_mat[triu_idx])) == len(dist_mat[triu_idx])

    # Assert purities are all distinct
    np.fill_diagonal(dist_mat, -np.inf)
    edges = []
    for k in (3, 5):
        for i in range(n):
            edges.append(np.argsort(dist_mat[i])[:k])
    purity = np.zeros(n)
    degree = np.zeros(n)
    for edge in edges:
        sub = labels[edge]
        for j in range(len(edge)):
            degree[edge[j]] += 1
            purity[edge[j]] += np.sum(sub[j] == sub) / len(edge)
    purity = purity / degree
    assert len(np.unique(purity)) == n

    ref_selected = _ref_hypergraph_prune_plain_argsort(sh, labels, save_rate=0.5)
    our_selected = _hypergraph_prune(sh, 11, labels, save_rate=0.5)
    np.testing.assert_array_equal(our_selected, ref_selected)


@pytest.mark.parametrize("threshold", [0.9, 0.5, 0.2])
def test_jaccard_similar_neighbors_matches_dense(threshold):
    """Sparse neighbours must equal pairs of dense Jaccard matrix above threshold."""
    rng = np.random.RandomState(0)
    n = 60
    sub = rng.normal(size=(n, 11))
    sub[30:] = sub[:30]  # duplicates create many identical hyperedge sets
    edges = _knn_hyperedges(sub, (3, 5))
    members = np.zeros((n, len(edges)), dtype=int)
    for e, nodes in enumerate(edges):
        members[nodes, e] = 1
    inter = members @ members.T
    union = members.sum(1)[:, None] + members.sum(1)[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros((n, n)), where=union != 0)
    np.fill_diagonal(jac, 0.0)
    similar = _jaccard_similar_neighbors(edges, n, threshold)
    for i in range(n):
        expected = np.where(jac[i] >= threshold)[0]
        assert sorted(similar[i].tolist()) == expected.tolist()


def _ref_knn_hyperedges_full_matrix(cleaned_sub, valid_k):
    """Compute reference full-matrix k-NN hyperedges with default argsort."""
    from sklearn.metrics.pairwise import euclidean_distances

    n = cleaned_sub.shape[0]
    if len(valid_k) == 0 or n == 0:
        return []
    dist_mat = euclidean_distances(cleaned_sub)
    np.fill_diagonal(dist_mat, -np.inf)
    edges_by_k = [[] for _ in valid_k]
    for i in range(n):
        order = np.argsort(dist_mat[i])
        for j, k in enumerate(valid_k):
            edges_by_k[j].append(order[:k].copy())
    return [edge for edges in edges_by_k for edge in edges]


@pytest.mark.parametrize("n", [50, 300])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("chunk_size", [1, 7, 64, 10000, None])
def test_knn_hyperedges_row_blocked_matches_full_matrix(n, seed, chunk_size):
    """Row-blocked k-NN must match full-matrix reference on tie-free continuous data."""
    from sklearn.metrics.pairwise import euclidean_distances

    rng = np.random.RandomState(seed)
    sub = rng.normal(size=(n, 11))

    dist_mat = euclidean_distances(sub)
    np.fill_diagonal(dist_mat, -np.inf)
    for i in range(n):
        s_dist = np.sort(dist_mat[i][dist_mat[i] != -np.inf])
        assert np.min(np.diff(s_dist)) > 1e-9

    ref_edges = _ref_knn_hyperedges_full_matrix(sub, (3, 5))
    blocked_edges = _knn_hyperedges(sub, (3, 5), chunk_size=chunk_size)
    assert len(ref_edges) == len(blocked_edges)
    for r_edge, b_edge in zip(ref_edges, blocked_edges):
        np.testing.assert_array_equal(r_edge, b_edge)


def test_knn_hyperedges_edge_cases():
    """Verify edge cases for row-blocked k-NN computation."""
    sub = np.random.RandomState(0).normal(size=(20, 11))
    assert _knn_hyperedges(sub, []) == []
    assert _knn_hyperedges(np.empty((0, 11)), [3]) == []

    out_large = _knn_hyperedges(sub, [50], chunk_size=7)
    assert len(out_large) == 20
    assert all(len(e) == 20 for e in out_large)
    assert all(e[0] == i for i, e in enumerate(out_large))
    assert all(e.base is None for e in out_large)
    for i in range(len(out_large)):
        for j in range(i + 1, len(out_large)):
            assert not np.shares_memory(out_large[i], out_large[j])
        assert not np.shares_memory(out_large[i], sub)

    for invalid_chunk in [0, -5, "10", 3.5, False]:
        with pytest.raises(
            ValueError, match="chunk_size must be a positive integer or None"
        ):
            _knn_hyperedges(sub, [3], chunk_size=invalid_chunk)


def test_knn_hyperedges_duplicates_valid_and_repeatable():
    """On data with many exact duplicates, output must be valid and repeatable."""
    n = 100
    sh = np.random.RandomState(7).normal(size=(n, 11))
    sh[50:] = sh[:50]
    first = _knn_hyperedges(sh, (3, 5), chunk_size=16)
    second = _knn_hyperedges(sh, (3, 5), chunk_size=16)
    assert len(first) == len(second) == 2 * n
    for e1, e2 in zip(first, second):
        np.testing.assert_array_equal(e1, e2)
        assert e1[0] in range(n)
        assert len(np.unique(e1)) == len(e1)
        assert e1.min() >= 0 and e1.max() < n
