"""Tests for _pelt module."""

import warnings

import numpy as np
import pytest

from aeon.segmentation import PELTSegmenter
from aeon.testing.data_generation import make_example_segment_series


def test_pelt_mean_shift():
    """PELT with L2 cost recovers mean-shift changepoints."""
    n_timepoints = 300
    series, true_cps = make_example_segment_series(
        n_timepoints=n_timepoints,
        shift_type="mean",
        return_changepoints=True,
        random_state=42,
    )
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    assert abs(cps[0] - true_cps[0]) <= 5, f"First cp {cps[0]} not near {true_cps[0]}"
    assert abs(cps[1] - true_cps[1]) <= 5, f"Second cp {cps[1]} not near {true_cps[1]}"


def test_pelt_variance_shift():
    """PELT with normal cost detects variance changepoints."""
    n_timepoints = 300
    series, true_cps = make_example_segment_series(
        n_timepoints=n_timepoints,
        shift_type="variance",
        return_changepoints=True,
        random_state=7,
    )
    pelt = PELTSegmenter(cost="normal", penalty="bic", min_size=10)
    cps = pelt.fit_predict(series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    assert abs(cps[0] - true_cps[0]) <= 10, f"First cp {cps[0]} not near {true_cps[0]}"
    assert abs(cps[1] - true_cps[1]) <= 10, f"Second cp {cps[1]} not near {true_cps[1]}"


def test_pelt_multivariate():
    """PELT detects changepoints in multivariate input."""
    n_timepoints = 300
    n_channels = 3
    series, true_cps = make_example_segment_series(
        n_timepoints=n_timepoints,
        n_channels=n_channels,
        shift_type="mean",
        axis=1,
        return_changepoints=True,
        random_state=42,
    )
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(series, axis=1)

    assert isinstance(cps, np.ndarray)
    assert len(cps) >= 1
    assert any(abs(cp - true_cps[0]) <= 5 for cp in cps), f"No cp near {true_cps[0]}"


def test_pelt_no_changepoint():
    """PELT returns empty array for constant series."""
    X = np.ones(100)
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(X)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 0


def test_pelt_bic_vs_manual_penalty():
    """BIC auto-penalty and equivalent manual penalty yield identical CPs."""
    n_timepoints = 300
    series, _ = make_example_segment_series(
        n_timepoints=n_timepoints,
        shift_type="mean",
        return_changepoints=True,
        random_state=42,
    )
    diff = np.diff(series)
    mad = float(np.median(np.abs(diff - np.median(diff))))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    manual_pen = float(2.0 * (sigma**2) * np.log(len(series)))

    pelt_bic = PELTSegmenter(cost="l2", penalty="bic")
    pelt_manual = PELTSegmenter(cost="l2", penalty=manual_pen)

    cps_bic = pelt_bic.fit_predict(series)
    cps_manual = pelt_manual.fit_predict(series)

    np.testing.assert_array_equal(cps_bic, cps_manual)


def test_pelt_jump():
    """Changepoints are still found with jump > 1."""
    n_timepoints = 300
    series, true_cps = make_example_segment_series(
        n_timepoints=n_timepoints,
        shift_type="mean",
        return_changepoints=True,
        random_state=42,
    )
    jump = 5
    pelt = PELTSegmenter(cost="l2", penalty="bic", jump=jump)
    cps = pelt.fit_predict(series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    assert cps[0] % jump == 0
    assert cps[1] % jump == 0
    assert abs(cps[0] - true_cps[0]) <= 10
    assert abs(cps[1] - true_cps[1]) <= 10


def test_pelt_short_series():
    """Series shorter than 2 * min_size returns empty array."""
    X = np.array([1.0, 2.0, 3.0])
    pelt = PELTSegmenter(cost="l2", penalty="bic", min_size=2)
    cps = pelt.fit_predict(X)
    assert len(cps) == 0


def test_pelt_none_and_zero_penalty():
    """penalty=None and penalty=0.0 run without error and return identical CPs."""
    X = np.array([1.0, 1.0, 5.0, 5.0, 1.0, 1.0])
    pelt_none = PELTSegmenter(cost="l2", penalty=None, min_size=2)
    pelt_zero = PELTSegmenter(cost="l2", penalty=0.0, min_size=2)
    cps_none = pelt_none.fit_predict(X)
    cps_zero = pelt_zero.fit_predict(X)
    np.testing.assert_array_equal(cps_none, cps_zero)


def test_pelt_aic_penalty():
    """AIC penalty matches exact manual penalty formula."""
    series, _ = make_example_segment_series(
        n_timepoints=300, shift_type="mean", return_changepoints=True, random_state=42
    )
    diff = np.diff(series)
    mad = float(np.median(np.abs(diff - np.median(diff))))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    manual_pen_l2 = float(4.0 * (sigma**2))

    pelt_aic = PELTSegmenter(cost="l2", penalty="aic")
    pelt_manual = PELTSegmenter(cost="l2", penalty=manual_pen_l2)
    np.testing.assert_array_equal(
        pelt_aic.fit_predict(series), pelt_manual.fit_predict(series)
    )

    pelt_norm_aic = PELTSegmenter(cost="normal", penalty="aic", min_size=10)
    pelt_norm_manual = PELTSegmenter(cost="normal", penalty=4.0, min_size=10)
    np.testing.assert_array_equal(
        pelt_norm_aic.fit_predict(series), pelt_norm_manual.fit_predict(series)
    )


@pytest.mark.parametrize("cost", ["l2", "normal"])
@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0, 1e-5])
def test_pelt_scale_invariance(cost, scale):
    """PELT changepoints are invariant to data scaling across orders of magnitude."""
    shift = "mean" if cost == "l2" else "variance"
    min_size = 2 if cost == "l2" else 10
    base_series = make_example_segment_series(
        n_timepoints=300, shift_type=shift, random_state=42
    )
    pelt = PELTSegmenter(cost=cost, penalty="bic", min_size=min_size)
    cps_base = pelt.fit_predict(base_series)
    cps_scaled = pelt.fit_predict(base_series * scale)
    np.testing.assert_array_equal(cps_scaled, cps_base)


@pytest.mark.parametrize(
    "kwargs,X,expected_msg",
    [
        ({"cost": "invalid"}, np.ones(20), "Invalid cost"),
        ({"penalty": "unknown"}, np.ones(20), "Invalid penalty"),
        (
            {"penalty": -1.0},
            np.ones(20),
            "Manual penalty must be a non-negative number",
        ),
        ({"penalty": True}, np.ones(20), "penalty cannot be a boolean"),
        ({"penalty": False}, np.ones(20), "penalty cannot be a boolean"),
        ({"penalty": np.nan}, np.ones(20), "penalty must be finite"),
        ({"penalty": np.inf}, np.ones(20), "penalty must be finite"),
        ({"penalty": -np.inf}, np.ones(20), "penalty must be finite"),
        ({"min_size": 1}, np.ones(20), "min_size must be an integer >= 2"),
        ({"min_size": True}, np.ones(20), "min_size cannot be a boolean"),
        ({"min_size": False}, np.ones(20), "min_size cannot be a boolean"),
        ({"jump": 0}, np.ones(20), "jump must be an integer >= 1"),
        ({"jump": True}, np.ones(20), "jump cannot be a boolean"),
        ({"jump": False}, np.ones(20), "jump cannot be a boolean"),
        ({}, np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0]), "contains infinite values"),
        (
            {},
            np.array([1.0, -np.inf, 3.0, 4.0, 5.0, 6.0]),
            "contains infinite values",
        ),
    ],
)
def test_pelt_validation_errors(kwargs, X, expected_msg):
    """Test that invalid parameters and non-finite inputs raise ValueError."""
    pelt = PELTSegmenter(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match=expected_msg):
            pelt.fit_predict(X)


@pytest.mark.parametrize("int_type", [int, np.int32, np.int64])
def test_pelt_accepts_integer_types(int_type):
    """PELTSegmenter accepts standard int and np.integer for min_size and jump."""
    X = np.ones(20)
    pelt = PELTSegmenter(min_size=int_type(3), jump=int_type(2))
    cps = pelt.fit_predict(X)
    assert isinstance(cps, np.ndarray)


# ---- brute-force exact partition solver & tests ----


def _exact_partition_solver(X, cost="l2", penalty=5.0, min_size=2, jump=1, axis=1):
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
    eps = np.empty(n_channels, dtype=np.float64)
    for c in range(n_channels):
        c_var = float(np.var(data[c]))
        eps[c] = 1e-8 * c_var if c_var > 0.0 else 1e-8

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
            return float(np.sum(length * np.log(var + eps)))

    for t in grid[1:]:
        best_cost = np.inf
        best_tau = 0
        for tau in grid:
            if tau < t and (t - tau) >= min_size and tau in F and np.isfinite(F[tau]):
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
