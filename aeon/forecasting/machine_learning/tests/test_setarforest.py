"""Test the setar-forest forecaster."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from aeon.forecasting.machine_learning._setarforest import SETARForest


def _make_series_with_exog(T=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.sin(2 * np.pi * t / 12.0)
    y = base + 0.1 * rng.standard_normal(T)  # univariate target
    exog = np.stack(
        [
            0.5 * base + 0.1 * rng.standard_normal(T),
            np.cos(2 * np.pi * t / 12.0) + 0.1 * rng.standard_normal(T),
        ],
        axis=1,
    )
    return y, exog


def test_setar_forest_runs_and_averages_with_exog():
    """Fit and predict with univariate y and exogenous features."""
    y, exog = _make_series_with_exog(T=120, seed=0)
    f = SETARForest(
        lag=10,
        n_estimators=5,
        bagging_fraction=0.7,
        feature_fraction=0.6,
        max_depth=5,
        random_state=123,
    )
    f.fit(y, exog=exog)
    p = f.predict(y, exog=exog)
    assert np.isfinite(p)


def test_setar_forest_deterministic_with_random_state():
    """Predictions are deterministic when random_state is fixed."""
    y, exog = _make_series_with_exog(T=100, seed=1)
    f1 = SETARForest(
        lag=10,
        n_estimators=4,
        bagging_fraction=0.75,
        feature_fraction=0.5,
        max_depth=4,
        random_state=42,
    )
    f2 = SETARForest(
        lag=10,
        n_estimators=4,
        bagging_fraction=0.75,
        feature_fraction=0.5,
        max_depth=4,
        random_state=42,
    )
    f1.fit(y, exog=exog)
    f2.fit(y, exog=exog)
    p1 = f1.predict(y, exog=exog)
    p2 = f2.predict(y, exog=exog)
    assert np.isfinite(p1) and np.isfinite(p2)
    assert p1 == pytest.approx(p2)

    f3 = SETARForest(
        lag=10,
        n_estimators=4,
        bagging_fraction=0.75,
        feature_fraction=0.5,
        max_depth=4,
        random_state=7,
    )
    f3.fit(y, exog=exog)
    p3 = f3.predict(y, exog=exog)
    assert np.isfinite(p3)


def test_setar_forest_integer_conversion_rounds_prediction():
    """integer_conversion rounds the averaged prediction."""
    y, exog = _make_series_with_exog(T=96, seed=2)
    kw = dict(
        lag=10,
        n_estimators=3,
        bagging_fraction=0.8,
        feature_fraction=0.7,
        max_depth=3,
        random_state=999,
    )
    f_raw = SETARForest(**kw, integer_conversion=False).fit(y, exog=exog)
    f_int = SETARForest(**kw, integer_conversion=True).fit(y, exog=exog)
    pr = f_raw.predict(y, exog=exog)
    pi = f_int.predict(y, exog=exog)
    assert np.isfinite(pr) and np.isfinite(pi)
    assert pi == pytest.approx(np.rint(pr))


def _embed_y_for_lr(y: np.ndarray, lag: int):
    """Build (targets, features) embedding for LR baseline compatible with tree."""
    y = np.asarray(y, dtype=float)
    rows = []
    for i in range(lag, len(y)):
        target = y[i]
        lags = y[i - lag : i][::-1]
        rows.append(np.concatenate(([target], lags)))
    E = np.asarray(rows)
    return E[:, 0], E[:, 1:]


def _final_lags_vector(y: np.ndarray, lag: int):
    return y[-lag:][::-1].reshape(1, -1)


def test_forest_matches_linear_regression_when_no_splits():
    """With max_depth=0 and one tree, prediction matches global linear regression."""
    rng = np.random.default_rng(123)
    T, lag = 200, 5
    e = 0.05 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = 0.7 * y[t - 1] - 0.2 * y[t - 2] + e[t]

    yt, X = _embed_y_for_lr(y, lag)
    lr = LinearRegression(fit_intercept=False).fit(X, yt)
    expected = float(lr.predict(_final_lags_vector(y, lag))[0])

    f = SETARForest(
        lag=lag,
        n_estimators=1,
        bagging_fraction=1.0,
        feature_fraction=1.0,
        max_depth=0,
        random_state=0,
    )
    f.fit(y)
    pred = f.predict(y)
    assert np.isfinite(pred)
    assert pred == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_forest_prediction_equals_mean_of_base_tree_predictions():
    """Forest prediction equals the mean of the base trees' predictions."""
    rng = np.random.default_rng(7)
    T, lag = 150, 8
    y = np.sin(2 * np.pi * np.arange(T) / 12.0) + 0.1 * rng.standard_normal(T)

    f = SETARForest(
        lag=lag,
        n_estimators=5,
        bagging_fraction=0.7,
        feature_fraction=0.6,
        max_depth=4,
        random_state=42,
    )
    f.fit(y)

    forest_pred = f.predict(y)
    base_preds = np.array([est.predict(y) for est in f.estimators_], dtype=float)
    mean_base = float(np.mean(base_preds))

    assert np.isfinite(forest_pred) and np.all(np.isfinite(base_preds))
    assert forest_pred == pytest.approx(mean_base, rel=1e-12, abs=1e-12)


def test_setar_forest_bagging_and_feature_subset_sizes_numpy():
    """Feature subsets are recorded and have valid sizes (NumPy indices)."""
    y, exog = _make_series_with_exog(T=110, seed=4)
    f = SETARForest(
        lag=10,
        n_estimators=3,
        bagging_fraction=0.6,
        feature_fraction=0.5,
        max_depth=3,
        random_state=1234,
    )
    f.fit(y, exog=exog)

    assert f.feature_subsets_, "feature_subsets_ should be populated"
    for subset in f.feature_subsets_:
        assert 1 <= len(subset) <= f.lag
        assert all(isinstance(i, int) and 0 <= i < f.lag for i in subset)
