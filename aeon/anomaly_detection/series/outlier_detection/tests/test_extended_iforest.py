"""Tests for the ExtendedIsolationForest class."""

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.outlier_detection import ExtendedIsolationForest
from aeon.utils.windowing import reverse_windowing, sliding_windows


def test_extended_iforest_default():
    """Test ExtendedIsolationForest univariate.

    The injected shift is -3 rather than -2: at -2 the anomaly is close enough to the
    tail of the normal data that the peak location depends on the random seed (it
    holds for roughly 31-37 of 40 seeds for either the current or the previous
    splitting rule), whereas -3 holds for 40/40 for both.
    """
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 3

    eif = ExtendedIsolationForest(window_size=10, stride=1, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_multivariate():
    """Test ExtendedIsolationForest multivariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80, 2))
    series[50:58, 0] -= 2

    eif = ExtendedIsolationForest(window_size=10, stride=1, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_no_window_univariate():
    """Test ExtendedIsolationForest without windows univariate."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    eif = ExtendedIsolationForest(window_size=1, stride=1, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_stride():
    """Test ExtendedIsolationForest with stride.

    Uses a -3 shift for the same seed-robustness reason as
    ``test_extended_iforest_default``.
    """
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 3

    eif = ExtendedIsolationForest(window_size=10, stride=2, random_state=0)
    pred = eif.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_semi_supervised():
    """Test ExtendedIsolationForest fit on normal data then predict."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2
    train_series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(window_size=10, stride=1, random_state=0)
    eif.fit(train_series, axis=0)
    pred = eif.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float64
    assert 50 <= np.argmax(pred) <= 60


def test_extended_iforest_deterministic():
    """Same random_state must give identical scores."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    a = ExtendedIsolationForest(window_size=10, random_state=42).fit_predict(
        series, axis=0
    )
    b = ExtendedIsolationForest(window_size=10, random_state=42).fit_predict(
        series, axis=0
    )
    assert np.array_equal(a, b)


def test_extended_iforest_extension_level_zero_matches_sklearn_scores():
    """``extension_level=0`` must reduce to the axis-parallel Isolation Forest.

    Reduction to the standard Isolation Forest is a correctness property of EIF, so
    this compares the anomaly *scores* against scikit-learn's ``IsolationForest`` by
    rank correlation on identical windows, rather than only checking that both
    detectors happen to locate the injected anomaly.
    """
    rng = check_random_state(0)
    series = rng.normal(size=(300,))
    series[150:158] -= 4
    window_size = 8

    windows, padding = sliding_windows(
        series, window_size=window_size, stride=1, axis=0
    )
    iforest = IsolationForest(
        n_estimators=200, max_samples=min(256, len(windows)), random_state=0
    ).fit(windows)
    # scikit-learn returns higher values for more normal points; negate to match
    # the aeon convention of higher score = more anomalous.
    sklearn_scores = reverse_windowing(
        -iforest.score_samples(windows), window_size, np.nanmean, 1, padding
    )

    axis_parallel = ExtendedIsolationForest(
        window_size=window_size, extension_level=0, n_estimators=200, random_state=0
    ).fit_predict(series, axis=0)

    assert spearmanr(axis_parallel, sklearn_scores).statistic > 0.98


def test_extended_iforest_single_subsample_is_not_nan():
    """A sub-sample of one window must not produce NaN scores.

    ``c(1) == 0``, so the ``mean_path / c`` normalisation is 0/0. Every point is
    trivially isolated at depth zero and carries no ranking information, so the
    scores fall back to the neutral value 0.5 (as scikit-learn's IsolationForest
    does when its denominator is zero).
    """
    rng = check_random_state(0)
    series = rng.normal(size=(80,))
    series[50:58] -= 2

    pred = ExtendedIsolationForest(
        window_size=10, max_samples=1, n_estimators=5, random_state=0
    ).fit_predict(series, axis=0)

    assert not np.isnan(pred).any()
    assert np.allclose(pred, 0.5)


def test_extended_iforest_invalid_extension_level():
    """extension_level outside [0, n_features - 1] must raise."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(window_size=10, extension_level=999, random_state=0)
    with pytest.raises(ValueError, match="extension_level must be between"):
        eif.fit_predict(series, axis=0)


def test_extended_iforest_invalid_max_samples():
    """A string max_samples other than 'auto' must raise."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(window_size=10, max_samples="all", random_state=0)
    with pytest.raises(ValueError, match="max_samples"):
        eif.fit_predict(series, axis=0)


def test_extended_iforest_invalid_n_estimators():
    """n_estimators must be a positive integer."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(window_size=10, n_estimators=0, random_state=0)
    with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
        eif.fit_predict(series, axis=0)


@pytest.mark.parametrize("max_samples", [0, -5, 0.0, 1.5])
def test_extended_iforest_invalid_max_samples_value(max_samples):
    """Out-of-range int and float max_samples must raise."""
    rng = check_random_state(0)
    series = rng.normal(size=(80,))

    eif = ExtendedIsolationForest(
        window_size=10, max_samples=max_samples, random_state=0
    )
    with pytest.raises(ValueError, match="max_samples"):
        eif.fit_predict(series, axis=0)


# ---------------------------------------------------------------------------
# Direct validation against the authors' reference implementation.
#
# ``_reference_*`` below is a transcription of the pure-Python ``eif_old.py``
# from the Extended Isolation Forest authors' repository
# (https://github.com/sahandha/eif, BSD-2-Clause), reduced to the parts that
# determine path lengths and kept deliberately literal so it can be diffed
# against the original. It is only used to check our implementation and is not
# part of the estimator.
# ---------------------------------------------------------------------------


def _reference_c_factor(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + np.euler_gamma) - 2.0 * (n - 1) / n


def _reference_tree(X, e, limit, exlevel, rng):
    if e >= limit or len(X) <= 1:
        return {"exnode": True, "size": len(X)}
    dim = X.shape[1]
    mins, maxs = X.min(axis=0), X.max(axis=0)
    idxs = rng.choice(range(dim), dim - exlevel - 1, replace=False)
    n = rng.normal(0, 1, dim)
    n[idxs] = 0
    p = rng.uniform(mins, maxs)
    w = (X - p).dot(n) < 0
    return {
        "exnode": False,
        "n": n,
        "p": p,
        "left": _reference_tree(X[w], e + 1, limit, exlevel, rng),
        "right": _reference_tree(X[~w], e + 1, limit, exlevel, rng),
    }


def _reference_path(x, node, e=0):
    if node["exnode"]:
        if node["size"] <= 1:
            return e
        return e + _reference_c_factor(node["size"])
    if (x - node["p"]).dot(node["n"]) < 0:
        return _reference_path(x, node["left"], e + 1)
    return _reference_path(x, node["right"], e + 1)


def _reference_scores(X, n_trees, sample, exlevel, seed):
    rng = np.random.RandomState(seed)
    limit = int(np.ceil(np.log2(sample)))
    trees = [
        _reference_tree(
            X[rng.choice(len(X), sample, replace=False)], 0, limit, exlevel, rng
        )
        for _ in range(n_trees)
    ]
    mean_path = np.array(
        [np.mean([_reference_path(x, tree) for tree in trees]) for x in X]
    )
    return 2.0 ** (-mean_path / _reference_c_factor(sample))


def _outlier_cluster(seed):
    """Return a Gaussian cloud with a compact cluster of outliers."""
    rng = check_random_state(seed)
    X = rng.normal(size=(300, 4))
    X[:20] += 6.0
    return X


def _two_clusters(seed):
    """Return two separated blobs, the case that motivates hyperplane splits."""
    rng = check_random_state(seed)
    X = np.vstack(
        [
            rng.normal(loc=[0, 0, 0, 0], scale=1.0, size=(150, 4)),
            rng.normal(loc=[6, 6, 0, 0], scale=1.0, size=(150, 4)),
        ]
    )
    # A point in the "shadow" region that axis-parallel splitting scores wrongly.
    X[0] = [3, 3, 4, 4]
    return X


def _level_shift(seed):
    """Return data with a sustained shift in one channel."""
    rng = check_random_state(seed)
    X = rng.normal(size=(300, 4))
    X[150:158, 0] -= 5.0
    return X


@pytest.mark.parametrize(
    "dataset", [_outlier_cluster, _two_clusters, _level_shift], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("extension_level", [0, 2, 3])
def test_extended_iforest_matches_reference_implementation(dataset, extension_level):
    """Scores must agree with the authors' ``eif_old.py`` at every extension level.

    The comparison against scikit-learn only pins down the ``extension_level=0``
    limit, so this checks the extended cases against the reference algorithm
    itself: level ``0`` (axis-parallel), an intermediate level, and the fully
    extended level, on three fixed datasets.

    Both forests are Monte-Carlo estimates with independent randomness, so the
    scores cannot be compared exactly; the assertion is on the size of the gap.
    ``0.05`` sits well above the spread seen here (worst case ``0.039`` over 45
    dataset/level/seed combinations) and below what a change to the splitting
    rule costs: reinstating the degenerate-split early return that this
    implementation deliberately drops takes the worst case to ``0.072`` and
    fails six of the nine cases below, every one of them at an extended level.
    """
    X = dataset(0)
    sample = min(256, len(X))

    reference = _reference_scores(
        X, n_trees=200, sample=sample, exlevel=extension_level, seed=0
    )
    scores = ExtendedIsolationForest(
        window_size=1,
        extension_level=extension_level,
        n_estimators=200,
        random_state=0,
    ).fit_predict(X, axis=0)

    assert np.max(np.abs(scores - reference)) < 0.05
    assert spearmanr(scores, reference).statistic > 0.95
