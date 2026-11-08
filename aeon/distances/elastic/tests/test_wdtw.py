"""Tests for WDTW distance."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import wdtw_cost_matrix, wdtw_distance, wdtw_pairwise_distance
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
def test_wdtw_distance_mixed_dtypes(dtype_x, dtype_y, bounding):
    """wdtw_distance on series with different dtypes equals the cost matrix corner.

    The rolling row kernels put the longer series first, so the lengths cover no
    swap and a swap in each direction. Bounding parameters must raise a
    FutureWarning and their absence must not.
    """
    rng = np.random.default_rng(0)
    for n_channels in MIXED_CHANNELS:
        for x_size, y_size in MIXED_LENGTHS:
            x = _make_mixed((n_channels, x_size), dtype_x, rng)
            y = _make_mixed((n_channels, y_size), dtype_y, rng)

            dist = _call_bounded(wdtw_distance, x, y, bounding)
            cost_matrix = _call_bounded(wdtw_cost_matrix, x, y, bounding)

            # An infinite distance would agree while testing nothing.
            assert np.isfinite(dist)
            assert_allclose(
                dist, cost_matrix[-1, -1], rtol=_mixed_rtol(dtype_x, dtype_y)
            )


@pytest.mark.parametrize("dtype_y", [np.int64, np.float32])
@pytest.mark.parametrize("bounding", [{}, WINDOW])
def test_wdtw_pairwise_distance_mixed_dtypes(dtype_y, bounding):
    """wdtw_pairwise_distance on collections with different dtypes matches pairs.

    Each entry must equal the cost matrix corner for that pair of series, in
    both argument orders.
    """
    n_channels = MIXED_CHANNELS[-1]
    rng = np.random.default_rng(0)
    X = _make_mixed((N_CASES_X, n_channels, SHORT_LENGTH), np.float64, rng)
    Y = _make_mixed((N_CASES_Y, n_channels, LONG_LENGTH), dtype_y, rng)

    for a, b in [(X, Y), (Y, X)]:
        pw = _call_bounded(wdtw_pairwise_distance, a, b, bounding)
        expected = np.empty((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                expected[i, j] = _call_bounded(wdtw_cost_matrix, a[i], b[j], bounding)[
                    -1, -1
                ]
        assert_allclose(pw, expected, rtol=_mixed_rtol(dtype_y))
