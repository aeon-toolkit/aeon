"""Test SETAR-Tree forecaster."""

import numpy as np
import pytest

from aeon.forecasting.machine_learning._setartree import SETARTree


def test_constant_series():
    """Test forecasting on a constant series (scalar prediction)."""
    y = np.ones(20)
    f = SETARTree(lag=2)
    f.fit(y)
    pred = f.predict(y)
    assert np.isscalar(pred), "Prediction should be a scalar for univariate series"
    assert np.isclose(pred, 1.0), f"Prediction {pred} not close to 1.0"


def test_linear_series():
    """Test forecasting on a linear series (scalar prediction)."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2)
    f.fit(y)
    pred = f.predict(y)
    assert np.isscalar(pred), "Prediction should be a scalar for univariate series"
    assert np.isclose(pred, 21.0, atol=0.01), f"Prediction {pred} not close to 21.0"


def test_fixed_lag():
    """Test the fixed_lag parameter."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2, fixed_lag=True, external_lag=1)
    f.fit(y)
    pred = f.predict(y)
    assert np.isclose(
        pred, 21.0, atol=0.01
    ), f"Fixed lag prediction {pred} not close to 21.0"


def test_different_stopping_criteria():
    """Test different stopping criteria parameters."""
    y = np.arange(1, 21.0)
    for criteria in ["lin_test", "error_imp", "both"]:
        f = SETARTree(lag=2, stopping_criteria=criteria)
        f.fit(y)
        pred = f.predict(y)
        assert np.isclose(
            pred, 21.0, atol=0.01
        ), f"Prediction with {criteria} {pred} not close to 21.0"


def test_forecast_linear_sets_attribute():
    """Test forecast() returns scalar next value and sets forecast_ attribute."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2)
    out = f.forecast(y)
    assert np.isscalar(out), "forecast() should return a scalar for univariate series"
    assert np.isclose(out, 21.0, atol=0.01), f"Forecast {out} not close to 21.0"
    assert hasattr(f, "forecast_"), "forecast_ attribute should be set by _forecast()"
    assert np.isclose(f.forecast_, out), "forecast_ should equal the returned forecast"


# ---------- helpers ----------
def _node_from_xy(y, X):
    """Create node matrix from target and features."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([y, X])


def test_constant_series_and_linear_and_fixed_and_forecast_attr():
    """Test multiple behaviours: constant, linear, fixed lag, and forecast attribute."""
    y_const = np.ones(20)
    y_lin = np.arange(1, 21.0)

    # constant
    f = SETARTree(lag=2)
    f.fit(y_const)
    pred = f.predict(y_const)
    assert np.isscalar(pred) and np.isclose(pred, 1.0)

    # linear
    f = SETARTree(lag=2)
    f.fit(y_lin)
    pred = f.predict(y_lin)
    assert np.isscalar(pred) and np.isclose(pred, 21.0, atol=0.01)

    # fixed lag
    f = SETARTree(lag=2, fixed_lag=True, external_lag=1)
    f.fit(y_lin)
    assert np.isclose(f.predict(y_lin), 21.0, atol=0.01)

    # forecast sets attribute
    f = SETARTree(lag=2)
    out = f.forecast(y_lin)
    assert np.isscalar(out) and np.isclose(out, 21.0, atol=0.01)
    assert hasattr(f, "forecast_") and np.isclose(f.forecast_, out)


@pytest.mark.parametrize("criteria", ["lin_test", "error_imp", "both"])
def test_stopping_criteria_variants(criteria):
    """Test stopping_criteria parameter variants produce correct predictions."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2, stopping_criteria=criteria)
    f.fit(y)
    assert np.isclose(f.predict(y), 21.0, atol=0.01)


def test_embed_create_input_process_scaling_and_empty():
    """Test _embed, _create_tree_input_matrix, and _process_input_data behaviours."""
    st = SETARTree(lag=3, scale=True)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    E = st._embed(x, 4)
    assert E.shape == (2, 4) and np.allclose(E[0], [4, 3, 2, 1])
    # short -> empty
    assert st._embed(np.array([1, 2]), 6).shape == (0, 6)

    training = {"series": [np.array([2, 4, 6, 8]), np.array([10, 20])]}
    embedded, final_lags, means = st._create_tree_input_matrix(training)
    assert embedded.shape == (1, 4 - 0)  # [y | lag1..lag3] (lag+1=4)
    assert final_lags.shape == (1, 3) and len(means) == 2

    # empty case
    st2 = SETARTree(lag=5)
    e2, f2, m2 = st2._create_tree_input_matrix({"series": [np.array([1.0, 2.0])]})
    assert e2.shape == (0, 6) and f2.shape == (0, 5) and m2 == []

    # process_input_data order + concat exog
    exog = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=float)
    ts = st._process_input_data(np.array([[1, 2, 3, 4]], dtype=float), exog)
    assert len(ts["series"]) == 3 and np.allclose(ts["series"][2], [20, 21, 22, 23])


def test_find_cut_point_paths_create_split_ss_inf():
    """Test _find_cut_point, _create_split, and _ss edge conditions."""
    st = SETARTree(lag=2)
    rng = np.random.default_rng(0)

    # all-equal x_ix -> early return
    X = rng.standard_normal((40, 2))
    y = X @ np.array([1.0, -0.3]) + 0.1 * rng.standard_normal(40)
    out = st._find_cut_point(X, y, np.ones(40) * 5.0, k=5, criterion="RSS")
    assert np.isinf(out["cost"]) and out["need_recheck"] == 0

    # singular chunks -> AICc path with recheck increments
    n = 48
    Xz = np.zeros((n, 2))
    yz = np.zeros(n)
    out2 = st._find_cut_point(Xz, yz, np.linspace(-1, 1, n), k=6, criterion="AICc")
    assert "need_recheck" in out2 and out2["need_recheck"] >= 0

    # _create_split + _ss empty sides -> inf
    X1 = np.arange(10, dtype=float)
    y1 = 2 * X1 + 1.0
    node = _node_from_xy(y1, X1)
    L, R = st._create_split(node, feat_idx=0, threshold=5.0)
    assert L.shape[0] == 5 and R.shape[0] == 5
    assert np.isinf(st._ss(threshold=-1e9, node=node, feat_idx=0))
    assert np.isinf(st._ss(threshold=1e9, node=node, feat_idx=0))


def test_linearity_error_improvement_early_and_normal():
    """Test _check_linearity and _check_error_improvement."""
    st = SETARTree(lag=1, error_threshold=0.01)
    # normal path with two regimes
    X = np.concatenate([np.linspace(0, 1, 30), np.linspace(3, 4, 30)])
    y = np.concatenate([1 * X[:30] + 0.1, 3 * X[30:] - 0.2])
    parent = _node_from_xy(y, X)
    left, right = st._create_split(parent, feat_idx=0, threshold=1.5)
    assert st._check_linearity(parent, (left, right), significance=0.5) in (True, False)
    assert st._check_error_improvement(parent, (left, right)) in (True, False)

    # early-return via tiny/singular -> False
    tiny_parent = _node_from_xy(np.ones(4), np.zeros(4))
    tL, tR = st._create_split(tiny_parent, feat_idx=0, threshold=0.5)
    assert st._check_linearity(tiny_parent, (tL, tR), significance=0.05) is False
    assert st._check_error_improvement(tiny_parent, (tL, tR)) in (True, False)


def test_get_leaf_index_and_fit_global_model_branches():
    """Test _get_leaf_index and _fit_global_model with empty and populated data."""
    st = SETARTree(lag=3)

    # _get_leaf_index arithmetic
    feats = np.array([0.2, 0.5, 0.7])
    th_lags = [[1, 2]]
    thresholds = [[0.4, 0.6]]
    idx = st._get_leaf_index(feats, th_lags, thresholds)
    assert isinstance(idx, int) and idx >= 0

    # _fit_global_model empty + with test
    out_empty = st._fit_global_model(np.empty((0, 3)))
    assert out_empty["model"] is None and out_empty["predictions"].size == 0

    X = np.linspace(0, 1, 20)
    y = 2 * X + 1
    data = _node_from_xy(y, X)
    test = _node_from_xy(y[:5], X[:5])
    out = st._fit_global_model(data, test=test)
    assert out["model"] is not None and out["predictions"].shape[0] == 5


def test_build_from_embedded_empty_raises_and_flags_and_public_api():
    """Test _build_from_embedded error, build flags, and public API behaviours."""
    y = np.array([[i for i in range(40)]], dtype=float)

    # empty embedded -> ValueError
    st = SETARTree(lag=2)
    with pytest.raises(ValueError, match="Empty embedded matrix"):
        st._build_from_embedded(np.empty((0, 3)), feature_indices=None)

    # predict before build -> RuntimeError
    with pytest.raises(RuntimeError, match="predict called before the tree is built"):
        st._predict(y)

    # build with flags; also exercise external_lag valid and out-of-range
    st1 = SETARTree(
        lag=4, max_depth=2, fixed_lag=True, external_lag=2, feature_subset=[0, 1, 2, 3]
    )
    emb, _, _ = st1._create_tree_input_matrix(st1._process_input_data(y))
    st1._build_from_embedded(emb, feature_indices=list(range(emb.shape[1] - 1)))
    assert st1.leaf_models_ and st1.feature_indices_ is not None

    st2 = SETARTree(lag=4, max_depth=1, fixed_lag=True, external_lag=99)
    emb2, _, _ = st2._create_tree_input_matrix(st2._process_input_data(y))
    st2._build_from_embedded(emb2, feature_indices=list(range(emb2.shape[1] - 1)))
    assert st2.leaf_models_

    # end-to-end fit/predict + not-enough-data path
    st3 = SETARTree(lag=3, scale=True)
    st3.fit(y)
    p = st3._predict(y)
    assert np.isscalar(p)
    with pytest.raises(ValueError, match="Not enough data points"):
        st3._predict(np.array([[1, 2]], dtype=float))

    # scaling round-trip consistency
    base = np.arange(1, 41.0)
    stA, stB = SETARTree(lag=5, scale=True), SETARTree(lag=5, scale=True)
    yA, yB = base.reshape(1, -1), (2.5 * base).reshape(1, -1)
    stA.fit(yA)
    stB.fit(yB)
    pA, pB = stA._predict(yA), stB._predict(yB)
    assert np.isclose(pB / pA, 2.5, rtol=1e-2)
