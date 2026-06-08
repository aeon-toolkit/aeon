"""Tests for Manhattan distance functions.

This module tests manhattan_distance and manhattan_pairwise_distance,
validating correctness, mathematical properties, and edge case handling.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances.pointwise import (
    euclidean_distance,
    manhattan_distance,
    manhattan_pairwise_distance,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.testing_config import MULTITHREAD_TESTING


def test_manhattan_basic_correctness():
    """Test basic correctness with known values."""
    # Simple case:  |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    expected = 9.0
    result = manhattan_distance(x, y)
    assert_almost_equal(result, expected, decimal=10)

    # 2D case
    x_2d = np.array([[1.0, 2.0, 3.0]])
    y_2d = np.array([[4.0, 5.0, 6.0]])
    result_2d = manhattan_distance(x_2d, y_2d)
    assert_almost_equal(result_2d, expected, decimal=10)


def test_manhattan_identity():
    """Test that distance from series to itself is zero."""
    x = make_example_1d_numpy(15, random_state=42)
    result = manhattan_distance(x, x)
    assert_almost_equal(result, 0.0, decimal=10)

    # Multivariate
    x_multi = make_example_2d_numpy_series(15, 5, random_state=42)
    result_multi = manhattan_distance(x_multi, x_multi)
    assert_almost_equal(result_multi, 0.0, decimal=10)


def test_manhattan_symmetry():
    """Test that distance is symmetric."""
    x = make_example_1d_numpy(20, random_state=1)
    y = make_example_1d_numpy(20, random_state=2)

    d1 = manhattan_distance(x, y)
    d2 = manhattan_distance(y, x)

    assert_almost_equal(d1, d2, decimal=10)


def test_manhattan_non_negativity():
    """Test that distances are always non-negative."""
    for seed in [1, 10, 42, 99, 123]:
        x = make_example_1d_numpy(10, random_state=seed)
        y = make_example_1d_numpy(10, random_state=seed + 100)
        result = manhattan_distance(x, y)
        assert result >= 0


def test_manhattan_triangle_inequality():
    """Test triangle inequality holds for Manhattan distance."""
    x = make_example_1d_numpy(15, random_state=1)
    y = make_example_1d_numpy(15, random_state=2)
    z = make_example_1d_numpy(15, random_state=3)

    d_xz = manhattan_distance(x, z)
    d_xy = manhattan_distance(x, y)
    d_yz = manhattan_distance(y, z)

    assert d_xz <= d_xy + d_yz + 1e-10


def test_manhattan_vs_euclidean():
    """Test that Manhattan distance >= Euclidean distance."""
    x = make_example_1d_numpy(20, random_state=7)
    y = make_example_1d_numpy(20, random_state=13)

    manhattan_dist = manhattan_distance(x, y)
    euclidean_dist = euclidean_distance(x, y)

    # Manhattan (L1) >= Euclidean (L2) for all x, y
    assert manhattan_dist >= euclidean_dist - 1e-10


def test_manhattan_single_point():
    """Test with single point series."""
    x = np.array([5.0])
    y = np.array([10.0])
    result = manhattan_distance(x, y)
    assert_almost_equal(result, 5.0, decimal=10)


def test_manhattan_equal_series():
    """Test that equal series have zero distance."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = manhattan_distance(x, x)
    assert_almost_equal(result, 0.0, decimal=10)


def test_manhattan_multivariate():
    """Test with multivariate series."""
    x = make_example_2d_numpy_series(20, 5, random_state=1)
    y = make_example_2d_numpy_series(20, 5, random_state=2)

    result = manhattan_distance(x, y)
    assert isinstance(result, float)
    assert result > 0


def test_manhattan_different_channel_counts():
    """Test multivariate with different channel counts."""
    x = make_example_2d_numpy_series(10, 3, random_state=1)
    y = make_example_2d_numpy_series(10, 5, random_state=2)

    result = manhattan_distance(x, y)

    # Should use minimum channels
    y_truncated = y[:3, :]
    result_expected = manhattan_distance(x, y_truncated)
    assert_almost_equal(result, result_expected, decimal=10)


def test_manhattan_invalid_shape():
    """Test that invalid shapes raise ValueError."""
    x_1d = make_example_1d_numpy(10, random_state=1)
    y_2d = make_example_2d_numpy_series(10, 1, random_state=2)

    with pytest.raises(ValueError, match="x and y must be 1D or 2D"):
        manhattan_distance(x_1d, y_2d)


def test_manhattan_pairwise_self():
    """Test pairwise distance matrix to self."""
    X = make_example_3d_numpy(5, 1, 10, random_state=42, return_y=False)
    pw = manhattan_pairwise_distance(X)

    assert pw.shape == (5, 5)
    assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)
    assert np.allclose(pw, pw.T)
    assert np.all(pw >= 0)


def test_manhattan_pairwise_manual_comparison():
    """Test pairwise computation against manual calculation."""
    X = make_example_3d_numpy(4, 1, 8, random_state=1, return_y=False)
    pw = manhattan_pairwise_distance(X)

    manual_pw = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            manual_pw[i, j] = manhattan_distance(X[i], X[j])

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_manhattan_pairwise_multiple_to_multiple():
    """Test pairwise between two different collections."""
    X = make_example_3d_numpy(5, 1, 10, random_state=1, return_y=False)
    Y = make_example_3d_numpy(7, 1, 10, random_state=2, return_y=False)

    pw = manhattan_pairwise_distance(X, Y)
    assert pw.shape == (5, 7)
    assert np.all(pw >= 0)


def test_manhattan_pairwise_unequal_length():
    """Test pairwise with unequal length series."""
    X = make_example_3d_numpy_list(5, 1, random_state=1, return_y=False)
    pw = manhattan_pairwise_distance(X)

    assert pw.shape == (5, 5)
    assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_manhattan_pairwise_parallel(n_jobs):
    """Test parallel execution yields same result as serial."""
    X = make_example_3d_numpy(6, 1, 15, random_state=42, return_y=False)

    serial = manhattan_pairwise_distance(X, n_jobs=1)
    parallel = manhattan_pairwise_distance(X, n_jobs=n_jobs)

    assert_almost_equal(serial, parallel, decimal=10)


def test_manhattan_consistency_with_scipy():
    """Validate against scipy implementation."""
    try:
        from scipy.spatial.distance import cityblock

        x = make_example_1d_numpy(25, random_state=7)
        y = make_example_1d_numpy(25, random_state=13)

        result = manhattan_distance(x, y)
        expected = cityblock(x, y)

        assert_almost_equal(result, expected, decimal=10)
    except ImportError:
        pytest.skip("scipy not available")
