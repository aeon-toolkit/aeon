"""Tests for ERP distance."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import erp_cost_matrix, erp_distance, erp_pairwise_distance
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
def test_erp_distance_mixed_dtypes(dtype_x, dtype_y, bounding):
    """erp_distance on series with different dtypes equals the cost matrix corner.

    The rolling row kernels put the longer series first, so the lengths cover no
    swap and a swap in each direction. Bounding parameters must raise a
    FutureWarning and their absence must not.
    """
    rng = np.random.default_rng(0)
    for n_channels in MIXED_CHANNELS:
        for x_size, y_size in MIXED_LENGTHS:
            x = _make_mixed((n_channels, x_size), dtype_x, rng)
            y = _make_mixed((n_channels, y_size), dtype_y, rng)

            dist = _call_bounded(erp_distance, x, y, bounding)
            cost_matrix = _call_bounded(erp_cost_matrix, x, y, bounding)

            # An infinite distance would agree while testing nothing.
            assert np.isfinite(dist)
            assert_allclose(
                dist, cost_matrix[-1, -1], rtol=_mixed_rtol(dtype_x, dtype_y)
            )


@pytest.mark.parametrize("dtype_y", [np.int64, np.float32])
@pytest.mark.parametrize("bounding", [{}, WINDOW])
def test_erp_pairwise_distance_mixed_dtypes(dtype_y, bounding):
    """erp_pairwise_distance on collections with different dtypes matches pairs.

    Each entry must equal the cost matrix corner for that pair of series, in
    both argument orders.
    """
    n_channels = MIXED_CHANNELS[-1]
    rng = np.random.default_rng(0)
    X = _make_mixed((N_CASES_X, n_channels, SHORT_LENGTH), np.float64, rng)
    Y = _make_mixed((N_CASES_Y, n_channels, LONG_LENGTH), dtype_y, rng)

    for a, b in [(X, Y), (Y, X)]:
        pw = _call_bounded(erp_pairwise_distance, a, b, bounding)
        expected = np.empty((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                expected[i, j] = _call_bounded(erp_cost_matrix, a[i], b[j], bounding)[
                    -1, -1
                ]
        assert_allclose(pw, expected, rtol=_mixed_rtol(dtype_y))


def _erp_reference(x, y, g):
    """Recursive ERP definition: deleting a point costs its distance to g."""

    def erp(i, j):
        if i == len(x):
            return sum(abs(v - g) for v in y[j:])
        if j == len(y):
            return sum(abs(v - g) for v in x[i:])
        return min(
            erp(i + 1, j + 1) + abs(x[i] - y[j]),
            erp(i + 1, j) + abs(x[i] - g),
            erp(i, j + 1) + abs(y[j] - g),
        )

    return erp(0, 0)


def test_erp_matches_definition():
    """Leading gaps must cost the cumulative distance to g, not the total."""
    # delete the leading zeros of y (cost 0), match the 3s, delete the trailing zeros
    x = np.array([[3.0, 0.0, 0.0]])
    y = np.array([[0.0, 0.0, 3.0]])
    assert erp_distance(x, y) == 0.0
    assert erp_pairwise_distance(x[np.newaxis], y[np.newaxis])[0, 0] == 0.0
    assert erp_cost_matrix(x, y)[-1, -1] == 0.0

    rng = np.random.default_rng(0)
    for g in (0.0, 0.5):
        for n, m in [(1, 4), (5, 5), (6, 3)]:
            x = rng.normal(size=(1, n))
            y = rng.normal(size=(1, m))
            expected = _erp_reference(x[0], y[0], g)
            assert_allclose(erp_distance(x, y, g=g), expected)
            assert_allclose(erp_cost_matrix(x, y, g=g)[-1, -1], expected)


def test_erp_matches_definition_multichannel_g_arr():
    """Two channels with a per-channel g follow the same cumulative gap cost."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 5))
    y = rng.normal(size=(2, 4))
    g_arr = np.array([0.2, -0.3])

    def erp(i, j):
        if i == x.shape[1]:
            return sum(np.linalg.norm(y[:, k] - g_arr) for k in range(j, y.shape[1]))
        if j == y.shape[1]:
            return sum(np.linalg.norm(x[:, k] - g_arr) for k in range(i, x.shape[1]))
        return min(
            erp(i + 1, j + 1) + np.linalg.norm(x[:, i] - y[:, j]),
            erp(i + 1, j) + np.linalg.norm(x[:, i] - g_arr),
            erp(i, j + 1) + np.linalg.norm(y[:, j] - g_arr),
        )

    assert_allclose(erp_distance(x, y, g_arr=g_arr), erp(0, 0))
