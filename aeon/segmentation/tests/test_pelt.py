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
