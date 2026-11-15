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


def test_erp_leading_gaps_cost_cumulative():
    """Deleting the first i points costs the sum of their gap costs, not the total.

    Regression test for gh-4003: the first row and column of the cost matrix held
    the total gap cost of the whole series, so alignments starting with gaps were
    over-charged.
    """
    x = np.array([[3.0, 0.0, 0.0]])
    y = np.array([[0.0, 0.0, 3.0]])
    assert erp_distance(x, y) == 0.0
    assert erp_pairwise_distance(np.stack([x]), np.stack([y]))[0, 0] == 0.0

    # Cost matrix of a series against all zeros (g = 0): the only cost is deleting
    # x[:i], so row i is cumsum(x)[i].
    x = np.array([[1.0, 2.0, 4.0]])
    assert_allclose(
        erp_cost_matrix(x, np.zeros((1, 3))), [[1, 1, 1], [3, 3, 3], [7, 7, 7]]
    )


def test_erp_triangle_inequality():
    """ERP is a metric, so it must satisfy the triangle inequality (gh-4003)."""
    a, b, c = np.array([[1.0]]), np.array([[0.0, 1.0]]), np.array([[0.0, 2.0]])
    assert erp_distance(a, c) == 1.0
    assert erp_distance(a, c) <= erp_distance(a, b) + erp_distance(b, c)

    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b, c = (rng.normal(size=(1, rng.integers(1, 8))) for _ in range(3))
        assert erp_distance(a, c) <= erp_distance(a, b) + erp_distance(b, c) + 1e-9
