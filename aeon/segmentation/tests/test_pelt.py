"""Tests for _pelt module."""

import numpy as np
import pytest

from aeon.segmentation import PELTSegmenter
from aeon.testing.data_generation import (
    make_example_mean_shift_series,
    make_example_multivariate_series,
    make_example_variance_shift_series,
)


def test_pelt_mean_shift():
    """PELT with L2 cost recovers mean-shift changepoints."""
    series = make_example_mean_shift_series(random_state=42)
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    assert abs(cps[0] - 100) <= 5, f"First cp {cps[0]} not near 100"
    assert abs(cps[1] - 200) <= 5, f"Second cp {cps[1]} not near 200"


def test_pelt_variance_shift():
    """PELT with normal cost detects variance changepoints."""
    series = make_example_variance_shift_series(random_state=7)
    pelt = PELTSegmenter(cost="normal", penalty="bic", min_size=10)
    cps = pelt.fit_predict(series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    assert abs(cps[0] - 100) <= 10, f"First cp {cps[0]} not near 100"
    assert abs(cps[1] - 200) <= 10, f"Second cp {cps[1]} not near 200"


def test_pelt_multivariate():
    """PELT detects changepoints in multivariate input."""
    series = make_example_multivariate_series(random_state=42)
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(series, axis=1)

    assert isinstance(cps, np.ndarray)
    assert len(cps) >= 1
    assert any(abs(cp - 100) <= 5 for cp in cps), f"No changepoint near 100 in {cps}"


def test_pelt_no_changepoint():
    """PELT returns empty array for constant series."""
    X = np.ones(100)
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(X)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 0


def test_pelt_bic_vs_manual_penalty():
    """BIC auto-penalty and equivalent manual penalty yield identical CPs."""
    series = make_example_mean_shift_series(random_state=42)
    n = len(series)
    n_channels = 1
    manual_pen = n_channels * 1 * np.log(n)

    pelt_bic = PELTSegmenter(cost="l2", penalty="bic")
    pelt_manual = PELTSegmenter(cost="l2", penalty=manual_pen)

    cps_bic = pelt_bic.fit_predict(series)
    cps_manual = pelt_manual.fit_predict(series)

    np.testing.assert_array_equal(cps_bic, cps_manual)


def test_pelt_jump():
    """Changepoints are still found with jump > 1."""
    series = make_example_mean_shift_series(random_state=42)
    pelt = PELTSegmenter(cost="l2", penalty="bic", jump=5)
    cps = pelt.fit_predict(series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    assert cps[0] % 5 == 0
    assert cps[1] % 5 == 0
    assert abs(cps[0] - 100) <= 10
    assert abs(cps[1] - 200) <= 10


def test_pelt_short_series():
    """Series shorter than 2 * min_size returns empty array."""
    X = np.array([1.0, 2.0, 3.0])
    pelt = PELTSegmenter(cost="l2", penalty="bic", min_size=2)
    cps = pelt.fit_predict(X)
    assert len(cps) == 0


def test_pelt_invalid_cost():
    """Invalid cost string raises ValueError."""
    pelt = PELTSegmenter(cost="invalid")
    X = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="Invalid cost"):
        pelt.fit_predict(X)


def test_pelt_invalid_penalty_string():
    """Invalid penalty string raises ValueError."""
    pelt = PELTSegmenter(penalty="unknown")
    X = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="Invalid penalty"):
        pelt.fit_predict(X)


def test_pelt_invalid_penalty_value():
    """Negative penalty raises ValueError."""
    pelt = PELTSegmenter(penalty=-1.0)
    X = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="non-negative number"):
        pelt.fit_predict(X)


def test_pelt_none_and_zero_penalty():
    """penalty=None and penalty=0.0 run without error and return identical CPs."""
    X = np.array([1.0, 1.0, 5.0, 5.0, 1.0, 1.0])
    pelt_none = PELTSegmenter(cost="l2", penalty=None, min_size=2)
    pelt_zero = PELTSegmenter(cost="l2", penalty=0.0, min_size=2)
    cps_none = pelt_none.fit_predict(X)
    cps_zero = pelt_zero.fit_predict(X)
    np.testing.assert_array_equal(cps_none, cps_zero)


def test_pelt_invalid_min_size():
    """min_size < 2 raises ValueError."""
    pelt = PELTSegmenter(min_size=1)
    X = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="min_size"):
        pelt.fit_predict(X)


def test_pelt_invalid_jump():
    """Jump < 1 raises ValueError."""
    pelt = PELTSegmenter(jump=0)
    X = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="jump"):
        pelt.fit_predict(X)


def test_pelt_get_params():
    """get_params returns constructor arguments."""
    pelt = PELTSegmenter(cost="normal", penalty=5.0, min_size=3, jump=2)
    params = pelt.get_params()
    assert params["cost"] == "normal"
    assert params["penalty"] == 5.0
    assert params["min_size"] == 3
    assert params["jump"] == 2


def test_pelt_aic_penalty():
    """AIC penalty runs without error and returns changepoints."""
    series = make_example_mean_shift_series(random_state=42)
    pelt = PELTSegmenter(cost="l2", penalty="aic")
    cps = pelt.fit_predict(series)
    assert isinstance(cps, np.ndarray)
    assert len(cps) >= 2


# ---- brute-force exact partition solver & tests ----


def _exact_partition_solver(
    X, cost="l2", penalty=5.0, min_size=2, jump=1, axis=1
):
    """Brute-force exact optimal partition solver without pruning.

    Solves the optimal partitioning problem using unpruned dynamic
    programming over all valid segmentation boundaries.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_2d = X_arr[np.newaxis, :]
    elif axis == 0:
        X_2d = X_arr.T
    else:
        X_2d = X_arr

    n_channels, n_timepoints = X_2d.shape
    if n_timepoints < 2 * min_size:
        return np.array([], dtype=np.intp)

    data = X_2d - X_2d.mean(axis=1, keepdims=True)
    data_T = data.T
    S = np.zeros((n_timepoints + 1, n_channels), dtype=np.float64)
    SS = np.zeros((n_timepoints + 1, n_channels), dtype=np.float64)
    np.cumsum(data_T, axis=0, out=S[1:])
    np.cumsum(data_T**2, axis=0, out=SS[1:])

    grid = list(range(0, n_timepoints + 1, jump))
    if grid[-1] != n_timepoints:
        grid.append(n_timepoints)

    F = {0: -penalty}
    cp = {0: 0}

    def calc_cost(start, end):
        length = end - start
        diff_s = S[end] - S[start]
        diff_ss = SS[end] - SS[start]
        if cost == "l2":
            return float(np.sum(np.maximum(diff_ss - diff_s**2 / length, 0.0)))
        else:
            var = np.maximum(diff_ss / length - (diff_s / length) ** 2, 0.0)
            return float(np.sum(length * np.log(var + 1e-8)))

    for t in grid[1:]:
        best_cost = np.inf
        best_tau = 0
        for tau in grid:
            if (
                tau < t
                and (t - tau) >= min_size
                and tau in F
                and np.isfinite(F[tau])
            ):
                c = calc_cost(tau, t)
                total = F[tau] + c + penalty
                if total < best_cost:
                    best_cost = total
                    best_tau = tau
        F[t] = best_cost
        cp[t] = best_tau

    changepoints = []
    curr = n_timepoints
    while cp.get(curr, 0) != 0:
        changepoints.append(cp[curr])
        curr = cp[curr]
    changepoints.sort()
    return np.array(changepoints, dtype=np.intp)


def test_pelt_pruning_counterexample():
    """Test known counterexample where naive pruning drops the optimal split.

    For X = np.array([3.0, 9.0, 0.0, 2.0, 10.0]), cost='l2', penalty=5.0, min_size=2:
    The global optimum is no changepoints (cost 78.8 vs 79.0 for changepoint [2]).
    """
    X = np.array([3.0, 9.0, 0.0, 2.0, 10.0])
    pelt = PELTSegmenter(cost="l2", penalty=5.0, min_size=2)
    cps = pelt.fit_predict(X)
    expected = _exact_partition_solver(
        X, cost="l2", penalty=5.0, min_size=2, jump=1, axis=1
    )
    np.testing.assert_array_equal(expected, np.array([], dtype=np.intp))
    np.testing.assert_array_equal(cps, expected)


@pytest.mark.parametrize("cost", ["l2", "normal"])
@pytest.mark.parametrize("min_size", [2, 3])
@pytest.mark.parametrize("jump", [1, 2])
@pytest.mark.parametrize("axis", [0, 1])
def test_pelt_against_brute_force(cost, min_size, jump, axis):
    """Compare PELTSegmenter against the exact brute-force partition solver."""
    pelt = PELTSegmenter(cost=cost, penalty=3.0, min_size=min_size, jump=jump)

    for seed in range(20):
        rng = np.random.default_rng(seed)
        # 1D series
        if axis == 1:
            X_1d = rng.normal(size=12)
            if seed % 2 == 0:
                X_1d[6:] += 4.0
            cps_1d = pelt.fit_predict(X_1d, axis=1)
            expected_1d = _exact_partition_solver(
                X_1d, cost=cost, penalty=3.0, min_size=min_size, jump=jump, axis=1
            )
            np.testing.assert_array_equal(
                cps_1d,
                expected_1d,
                err_msg=(
                    f"1D failed on seed={seed}, cost={cost}, "
                    f"min_size={min_size}, jump={jump}"
                ),
            )

        # 2D series
        if axis == 1:
            X_2d = rng.normal(size=(2, 10))
            if seed % 2 == 0:
                X_2d[:, 5:] += 3.0
        else:
            X_2d = rng.normal(size=(10, 2))
            if seed % 2 == 0:
                X_2d[5:, :] += 3.0

        cps_2d = pelt.fit_predict(X_2d, axis=axis)
        expected_2d = _exact_partition_solver(
            X_2d, cost=cost, penalty=3.0, min_size=min_size, jump=jump, axis=axis
        )
        np.testing.assert_array_equal(
            cps_2d,
            expected_2d,
            err_msg=(
                f"2D failed on seed={seed}, cost={cost}, "
                f"min_size={min_size}, jump={jump}, axis={axis}"
            ),
        )
