"""Tests for MSM distance."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import msm_cost_matrix, msm_distance, msm_pairwise_distance
from aeon.distances.elastic._msm import _cost_dependent


def _reference_cost_dependent(x, y, z, c):
    """Previous NumPy formulation of the dependent MSM split/merge cost."""
    diameter = np.sum((y - z) ** 2)
    distance_to_mid = np.sum(((y + z) / 2.0 - x) ** 2)
    if distance_to_mid <= diameter / 4.0:
        return c
    dist_to_q_prev = np.sum((y - x) ** 2)
    dist_to_c = np.sum((z - x) ** 2)
    if dist_to_q_prev < dist_to_c:
        return c + dist_to_q_prev
    return c + dist_to_c


@pytest.mark.parametrize("n_channels", [1, 3, 10])
def test_cost_dependent_matches_reference(n_channels):
    """_cost_dependent is numerically equivalent to the previous formulation."""
    rng = np.random.default_rng(n_channels)
    for _ in range(200):
        x, y, z = rng.normal(scale=rng.uniform(0.1, 10.0), size=(3, n_channels))
        assert_allclose(
            _cost_dependent(x, y, z, 1.0),
            _reference_cost_dependent(x, y, z, 1.0),
            rtol=1e-12,
        )


def test_dependent_msm_rejects_different_channel_counts():
    """Dependent MSM raises a ValueError when channel counts differ."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 20))
    y = rng.normal(size=(2, 25))
    match = "same number of channels"

    with pytest.raises(ValueError, match=match):
        msm_distance(x, y, independent=False)
    with pytest.raises(ValueError, match=match):
        msm_cost_matrix(x, y, independent=False)
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match=match):
        msm_distance(x, y, independent=False, window=0.5)


MIXED_DTYPES = [
    (np.float64, np.int64),
    (np.int64, np.float64),
    (np.float32, np.float64),
    (np.float64, np.float32),
]
SHORT_LENGTH = 9
LONG_LENGTH = 14
# Equal lengths, then both directions of the internal swap to the longer series.
MIXED_LENGTHS = [
    (LONG_LENGTH, LONG_LENGTH),
    (SHORT_LENGTH, LONG_LENGTH),
    (LONG_LENGTH, SHORT_LENGTH),
]
MIXED_CHANNELS = [1, 3]
N_CASES_X = 3
N_CASES_Y = 2
WINDOW = {"window": 0.5}
ITAKURA = {"itakura_max_slope": 0.6}
# Spread the values so that integer series are not mostly zeros.
SCALE = 5
RTOL_DOUBLE = 1e-12
RTOL_SINGLE = 1e-6


def _make_mixed(shape, dtype, rng):
    """Random array of the given shape and dtype."""
    return (rng.normal(size=shape) * SCALE).astype(dtype)


def _mixed_rtol(*dtypes):
    """Tolerance for comparing kernels, looser if any input is single precision."""
    return RTOL_SINGLE if np.float32 in dtypes else RTOL_DOUBLE


def _call_msm(func, x, y, independent, bounding):
    """Call an MSM function, checking it warns only if bounding is requested."""
    if bounding:
        with pytest.warns(FutureWarning):
            return func(x, y, independent=independent, **bounding)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return func(x, y, independent=independent)


@pytest.mark.parametrize("dtype_x,dtype_y", MIXED_DTYPES)
@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("bounding", [{}, WINDOW, ITAKURA])
def test_msm_distance_mixed_dtypes(dtype_x, dtype_y, independent, bounding):
    """msm_distance on series with different dtypes equals the cost matrix corner.

    The rolling row kernels put the longer series first. Doing so by reassigning
    x and y fails Numba type unification when the dtypes differ, even when the
    series are equal length and no swap is needed.
    """
    rng = np.random.default_rng(0)
    for n_channels in MIXED_CHANNELS:
        for x_size, y_size in MIXED_LENGTHS:
            x = _make_mixed((n_channels, x_size), dtype_x, rng)
            y = _make_mixed((n_channels, y_size), dtype_y, rng)

            dist = _call_msm(msm_distance, x, y, independent, bounding)
            cost_matrix = _call_msm(msm_cost_matrix, x, y, independent, bounding)

            # An infinite distance would agree while testing nothing.
            assert np.isfinite(dist)
            assert_allclose(
                dist, cost_matrix[-1, -1], rtol=_mixed_rtol(dtype_x, dtype_y)
            )


@pytest.mark.parametrize("dtype_y", [np.int64, np.float32])
@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("bounding", [{}, WINDOW])
def test_msm_pairwise_distance_mixed_dtypes(dtype_y, independent, bounding):
    """msm_pairwise_distance on collections with different dtypes matches pairs.

    Each entry must equal the cost matrix corner for that pair of series, in
    both argument orders.
    """
    n_channels = MIXED_CHANNELS[-1]
    rng = np.random.default_rng(0)
    X = _make_mixed((N_CASES_X, n_channels, SHORT_LENGTH), np.float64, rng)
    Y = _make_mixed((N_CASES_Y, n_channels, LONG_LENGTH), dtype_y, rng)

    for a, b in [(X, Y), (Y, X)]:
        pw = _call_msm(msm_pairwise_distance, a, b, independent, bounding)
        expected = np.empty((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                expected[i, j] = _call_msm(
                    msm_cost_matrix, a[i], b[j], independent, bounding
                )[-1, -1]
        assert_allclose(pw, expected, rtol=_mixed_rtol(dtype_y))
