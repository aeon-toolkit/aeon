"""Tests for TWE distance."""

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose

from aeon.distances import twe_cost_matrix, twe_distance, twe_pairwise_distance
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic._twe import _pad_arrs
from aeon.distances.pointwise._euclidean import _univariate_euclidean_distance
from aeon.testing.utils._mixed_dtypes import (
    ITAKURA,
    LONG_LENGTH,
    MIXED_CHANNELS,
    MIXED_DTYPES,
    MIXED_LENGTHS,
    N_CASES_X,
    N_CASES_Y,
    SHORT_LENGTH,
    WINDOW,
    _call_bounded,
    _make_mixed,
    _mixed_rtol,
)


@pytest.mark.parametrize("dtype_x,dtype_y", MIXED_DTYPES)
@pytest.mark.parametrize("bounding", [{}, WINDOW, ITAKURA])
def test_twe_distance_mixed_dtypes(dtype_x, dtype_y, bounding):
    """twe_distance on series with different dtypes equals the cost matrix corner.

    The rolling row kernels put the longer series first, so the lengths cover no
    swap and a swap in each direction. Bounding parameters must raise a
    FutureWarning and their absence must not.
    """
    rng = np.random.default_rng(0)
    for n_channels in MIXED_CHANNELS:
        for x_size, y_size in MIXED_LENGTHS:
            x = _make_mixed((n_channels, x_size), dtype_x, rng)
            y = _make_mixed((n_channels, y_size), dtype_y, rng)

            dist = _call_bounded(twe_distance, x, y, bounding)
            cost_matrix = _call_bounded(twe_cost_matrix, x, y, bounding)

            # An infinite distance would agree while testing nothing.
            assert np.isfinite(dist)
            assert_allclose(
                dist, cost_matrix[-1, -1], rtol=_mixed_rtol(dtype_x, dtype_y)
            )


@pytest.mark.parametrize("dtype_y", [np.int64, np.float32])
@pytest.mark.parametrize("bounding", [{}, WINDOW])
def test_twe_pairwise_distance_mixed_dtypes(dtype_y, bounding):
    """twe_pairwise_distance on collections with different dtypes matches pairs.

    Each entry must equal the cost matrix corner for that pair of series, in
    both argument orders.
    """
    n_channels = MIXED_CHANNELS[-1]
    rng = np.random.default_rng(0)
    X = _make_mixed((N_CASES_X, n_channels, SHORT_LENGTH), np.float64, rng)
    Y = _make_mixed((N_CASES_Y, n_channels, LONG_LENGTH), dtype_y, rng)

    for a, b in [(X, Y), (Y, X)]:
        pw = _call_bounded(twe_pairwise_distance, a, b, bounding)
        expected = np.empty((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                expected[i, j] = _call_bounded(twe_cost_matrix, a[i], b[j], bounding)[
                    -1, -1
                ]
        assert_allclose(pw, expected, rtol=_mixed_rtol(dtype_y))


@njit(cache=True, fastmath=True)
def _previous_unbounded_twe(x, y, nu, lmbda):
    """Frozen pre-optimisation rolling-row recurrence, on padded float64 arrays."""
    x_size = x.shape[1]
    y_size = y.shape[1]
    del_add = nu + lmbda
    del_y_cost = np.zeros(y_size)
    for j in range(1, y_size):
        del_y_cost[j] = _univariate_euclidean_distance(y[:, j - 1], y[:, j]) + del_add
    prev = np.full(y_size, np.inf)
    curr = np.empty(y_size)
    prev[0] = 0.0
    for i in range(1, x_size):
        del_x_cost = _univariate_euclidean_distance(x[:, i - 1], x[:, i]) + del_add
        curr[0] = np.inf
        for j in range(1, y_size):
            del_x = prev[j] + del_x_cost
            del_y = curr[j - 1] + del_y_cost[j]
            match = (
                prev[j - 1]
                + _univariate_euclidean_distance(x[:, i], y[:, j])
                + _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1])
                + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
            )
            curr[j] = min(del_x, del_y, match)
        prev, curr = curr, prev
    return prev[y_size - 1]


def _previous_twe_distance(x, y, nu=0.001, lmbda=1.0):
    """Apply the original public padding and dimension-swap rules."""
    x = _pad_arrs(x.reshape(1, -1) if x.ndim == 1 else x)
    y = _pad_arrs(y.reshape(1, -1) if y.ndim == 1 else y)
    if x.shape[1] < y.shape[1]:
        return _previous_unbounded_twe(y, x, nu, lmbda)
    return _previous_unbounded_twe(x, y, nu, lmbda)


@pytest.mark.parametrize("n_channels", [1, 2, 5])
def test_twe_unbounded_randomised_equivalence(n_channels):
    """Preserve the original recurrence across 300 unequal-length random pairs."""
    rng = np.random.default_rng(3563 + n_channels)
    for _ in range(100):
        nx, ny = rng.integers(1, 40, size=2)
        x = rng.normal(size=(n_channels, nx))
        y = rng.normal(size=(n_channels, ny))
        nu = rng.choice([0.0, 0.001, 1.0, 10000.0])
        lmbda = rng.choice([0.0, 1.0, 25.0, 1000000.0])
        for a, b in [(x, y), (y, x)]:
            assert_allclose(
                twe_distance(a, b, nu=nu, lmbda=lmbda),
                _previous_twe_distance(a, b, nu, lmbda),
                rtol=1e-12,
                atol=0,
            )


def test_twe_unbounded_long_equivalence():
    """Distance reuse and scalar arithmetic preserve long-series accuracy."""
    rng = np.random.default_rng(4000)
    x, y = rng.normal(size=(2, 4000))
    assert_allclose(
        twe_distance(x, y), _previous_twe_distance(x, y), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("n_channels", [1, 2, 5])
@pytest.mark.parametrize("kind", ["integer", "constant", "identical"])
def test_twe_unbounded_special_series(n_channels, kind):
    """Preserve zero identity, constant series and integer-valued inputs."""
    rng = np.random.default_rng(7)
    x = rng.integers(-5, 6, size=(n_channels, 17))
    y = rng.integers(-5, 6, size=(n_channels, 11))
    if kind == "constant":
        x = np.full_like(x, -2)
        y = np.full_like(y, 3)
    elif kind == "identical":
        y = x.copy()
    result = twe_distance(x, y)
    assert_allclose(result, _previous_twe_distance(x, y), rtol=1e-12, atol=0)
    if kind == "identical":
        assert result == 0.0


@pytest.mark.parametrize("n_channels", [1, 2, 5])
@pytest.mark.parametrize("dtype_x,dtype_y", MIXED_DTYPES)
def test_twe_unbounded_mixed_precision_reference(n_channels, dtype_x, dtype_y):
    """Public padding keeps mixed input dtypes equivalent in both swap directions."""
    rng = np.random.default_rng(27)
    x = _make_mixed((n_channels, 9), dtype_x, rng)
    y = _make_mixed((n_channels, 19), dtype_y, rng)
    for a, b in [(x, y), (y, x)]:
        assert_allclose(
            twe_distance(a, b), _previous_twe_distance(a, b), rtol=1e-12, atol=0
        )


@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("unequal_length", [False, True])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_twe_unbounded_pairwise_reference(n_channels, unequal_length, n_jobs):
    """Both pairwise drivers preserve results with parallel and unequal-length input."""
    rng = np.random.default_rng(19)
    lengths = [7, 11, 5, 13, 9] if unequal_length else [9] * 5
    series = [rng.normal(size=(n_channels, length)) for length in lengths]
    x, y = series[:3], series[3:]
    for a, b in [(x, y), (y, x), (series, None)]:
        result = twe_pairwise_distance(a, b, n_jobs=n_jobs)
        target = a if b is None else b
        expected = np.array(
            [[_previous_twe_distance(xi, yj) for yj in target] for xi in a]
        )
        assert_allclose(result, expected, rtol=1e-12, atol=0)


@njit(cache=True, fastmath=True)
def _previous_twe_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float, lmbda: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = 0.0

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                # Deletion in x
                del_x_squared_dist = _univariate_euclidean_distance(
                    x[:, i - 1], x[:, i]
                )
                del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add
                # Deletion in y
                del_y_squared_dist = _univariate_euclidean_distance(
                    y[:, j - 1], y[:, j]
                )
                del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                # Match
                match_same_squared_d = _univariate_euclidean_distance(x[:, i], y[:, j])
                match_prev_squared_d = _univariate_euclidean_distance(
                    x[:, i - 1], y[:, j - 1]
                )
                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_squared_d
                    + match_prev_squared_d
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )

                cost_matrix[i, j] = min(del_x, del_y, match)

    return cost_matrix[1:, 1:]


@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("bounding", [{}, WINDOW, ITAKURA])
def test_twe_cost_matrix_precomputed_deletions(n_channels, bounding):
    """Hoisted deletion norms preserve the whole matrix, including bounded cells."""
    rng = np.random.default_rng(3564 + n_channels)
    for nx, ny in [(1, 1), (9, 14), (14, 9), (12, 12)]:
        if "itakura_max_slope" in bounding and min(nx, ny) < 3:
            continue
        for nu, lmbda in [(0.0, 0.0), (0.001, 1.0), (2.0, 10000.0)]:
            x = rng.normal(size=(n_channels, nx))
            y = rng.normal(size=(n_channels, ny))
            matrix = create_bounding_matrix(nx, ny, **bounding)
            expected = _previous_twe_cost_matrix(
                _pad_arrs(x), _pad_arrs(y), matrix, nu, lmbda
            )
            actual = _call_bounded(twe_cost_matrix, x, y, bounding, nu=nu, lmbda=lmbda)
            assert_allclose(actual, expected, rtol=1e-12, atol=0)
