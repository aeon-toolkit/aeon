"""Tests for _pelt module."""

import numpy as np
import pytest

from aeon.segmentation import PELTSegmenter

# ---- fixtures ----


@pytest.fixture
def mean_shift_series():
    """Three-segment univariate series with large mean shifts.

    Segments: mean=0 (100pts), mean=10 (100pts), mean=0 (100pts).
    Expected changepoints near 100 and 200.
    """
    rng = np.random.default_rng(42)
    seg1 = rng.normal(0, 0.5, 100)
    seg2 = rng.normal(10, 0.5, 100)
    seg3 = rng.normal(0, 0.5, 100)
    return np.concatenate([seg1, seg2, seg3])


@pytest.fixture
def variance_shift_series():
    """Three-segment univariate series with variance shifts only.

    Segments: var=1 (100pts), var=25 (100pts), var=1 (100pts).
    Mean is constant at 0.
    """
    rng = np.random.default_rng(7)
    seg1 = rng.normal(0, 1, 100)
    seg2 = rng.normal(0, 5, 100)
    seg3 = rng.normal(0, 1, 100)
    return np.concatenate([seg1, seg2, seg3])


@pytest.fixture
def multivariate_series():
    """Two-segment multivariate series (3 channels).

    Changepoint near index 100.
    """
    rng = np.random.default_rng(42)
    seg1 = rng.normal(0, 0.5, (3, 100))
    seg2 = rng.normal(8, 0.5, (3, 100))
    return np.hstack([seg1, seg2])


# ---- correctness tests ----


def test_pelt_mean_shift(mean_shift_series):
    """PELT with L2 cost recovers mean-shift changepoints."""
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(mean_shift_series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    # Changepoints should be near 100 and 200 (within ±5)
    assert abs(cps[0] - 100) <= 5, f"First cp {cps[0]} not near 100"
    assert abs(cps[1] - 200) <= 5, f"Second cp {cps[1]} not near 200"


def test_pelt_variance_shift(variance_shift_series):
    """PELT with normal cost detects variance changepoints."""
    pelt = PELTSegmenter(cost="normal", penalty="bic", min_size=10)
    cps = pelt.fit_predict(variance_shift_series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    assert abs(cps[0] - 100) <= 10, f"First cp {cps[0]} not near 100"
    assert abs(cps[1] - 200) <= 10, f"Second cp {cps[1]} not near 200"


def test_pelt_multivariate(multivariate_series):
    """PELT detects changepoints in multivariate input."""
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    # multivariate_series is already (3, 200)
    cps = pelt.fit_predict(multivariate_series, axis=1)

    assert isinstance(cps, np.ndarray)
    assert len(cps) >= 1
    # At least one changepoint should be near 100
    assert any(abs(cp - 100) <= 5 for cp in cps), f"No changepoint near 100 in {cps}"


def test_pelt_no_changepoint():
    """PELT returns empty array for constant series."""
    X = np.ones(100)
    pelt = PELTSegmenter(cost="l2", penalty="bic")
    cps = pelt.fit_predict(X)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 0


def test_pelt_bic_vs_manual_penalty(mean_shift_series):
    """BIC auto-penalty and equivalent manual penalty yield identical CPs."""
    n = len(mean_shift_series)
    n_channels = 1  # univariate
    manual_pen = n_channels * 1 * np.log(n)  # p=1 for L2

    pelt_bic = PELTSegmenter(cost="l2", penalty="bic")
    pelt_manual = PELTSegmenter(cost="l2", penalty=manual_pen)

    cps_bic = pelt_bic.fit_predict(mean_shift_series)
    cps_manual = pelt_manual.fit_predict(mean_shift_series)

    np.testing.assert_array_equal(cps_bic, cps_manual)


def test_pelt_jump(mean_shift_series):
    """Changepoints are still found with jump > 1."""
    pelt = PELTSegmenter(cost="l2", penalty="bic", jump=5)
    cps = pelt.fit_predict(mean_shift_series)

    assert isinstance(cps, np.ndarray)
    assert len(cps) == 2
    # With jump=5, changepoints snap to multiples of 5
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


# ---- parameter validation tests ----


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


# ---- estimator protocol ----


def test_pelt_get_params():
    """get_params returns constructor arguments."""
    pelt = PELTSegmenter(cost="normal", penalty=5.0, min_size=3, jump=2)
    params = pelt.get_params()
    assert params["cost"] == "normal"
    assert params["penalty"] == 5.0
    assert params["min_size"] == 3
    assert params["jump"] == 2


def test_pelt_aic_penalty(mean_shift_series):
    """AIC penalty runs without error and returns changepoints."""
    pelt = PELTSegmenter(cost="l2", penalty="aic")
    cps = pelt.fit_predict(mean_shift_series)
    assert isinstance(cps, np.ndarray)
    # AIC penalty is lighter than BIC → at least as many changepoints
    assert len(cps) >= 2
