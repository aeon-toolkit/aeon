"""Tests for Squared distance functions.

This module tests squared_distance and squared_pairwise_distance,
validating correctness and relationship to euclidean distance.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances.pointwise import (
    euclidean_distance,
    squared_distance,
    squared_pairwise_distance,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.testing_config import MULTITHREAD_TESTING


def test_squared_basic_correctness():
    """Test basic correctness with known values."""
    # (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    expected = 27.0
    result = squared_distance(x, y)
    assert_almost_equal(result, expected, decimal=10)


def test_squared_relationship_to_euclidean():
    """Test that squared equals euclidean squared."""
    x = make_example_1d_numpy(20, random_state=42)
    y = make_example_1d_numpy(20, random_state=24)

    squared_result = squared_distance(x, y)
    euclidean_result = euclidean_distance(x, y)

    assert_almost_equal(squared_result, euclidean_result**2, decimal=10)
    assert_almost_equal(np.sqrt(squared_result), euclidean_result, decimal=10)


def test_squared_identity():
    """Test that distance from series to itself is zero."""
    x = make_example_1d_numpy(15, random_state=42)
    result = squared_distance(x, x)
    assert_almost_equal(result, 0.0, decimal=10)


def test_squared_symmetry():
    """Test that distance is symmetric."""
    x = make_example_1d_numpy(20, random_state=1)
    y = make_example_1d_numpy(20, random_state=2)

    d1 = squared_distance(x, y)
    d2 = squared_distance(y, x)

    assert_almost_equal(d1, d2, decimal=10)


def test_squared_non_negativity():
    """Test that distances are always non-negative."""
    for seed in [1, 10, 42, 99, 123]:
        x = make_example_1d_numpy(10, random_state=seed)
        y = make_example_1d_numpy(10, random_state=seed + 100)
        result = squared_distance(x, y)
        assert result >= 0


def test_squared_triangle_inequality_does_not_hold():
    """Squared distance does NOT satisfy triangle inequality."""
    # Find a counterexample where d(x,z)^2 > d(x,y)^2 + d(y,z)^2
    # Example: x=0, y=1, z=2
    # d(x,z)^2 = 4, but d(x,y)^2 + d(y,z)^2 = 1 + 1 = 2
    # So 4 > 2, triangle inequality fails

    x = np.array([0.0])
    y = np.array([1.0])
    z = np.array([2.0])

    d_xz = squared_distance(x, z)  # 4
    d_xy = squared_distance(x, y)  # 1
    d_yz = squared_distance(y, z)  # 1

    # Triangle inequality would require d_xz <= d_xy + d_yz
    # But 4 > 2, so it doesn't hold
    assert d_xz > d_xy + d_yz


def test_squared_single_point():
    """Test with single point series."""
    x = np.array([5.0])
    y = np.array([10.0])
    result = squared_distance(x, y)
    assert_almost_equal(result, 25.0, decimal=10)


def test_squared_multivariate():
    """Test with multivariate series."""
    x = make_example_2d_numpy_series(20, 5, random_state=1)
    y = make_example_2d_numpy_series(20, 5, random_state=2)

    result = squared_distance(x, y)
    assert isinstance(result, float)
    assert result > 0


def test_squared_different_channel_counts():
    """Test multivariate with different channel counts."""
    x = make_example_2d_numpy_series(10, 3, random_state=1)
    y = make_example_2d_numpy_series(10, 5, random_state=2)

    result = squared_distance(x, y)

    # Should use minimum channels
    y_truncated = y[:3, :]
    result_expected = squared_distance(x, y_truncated)
    assert_almost_equal(result, result_expected, decimal=10)


def test_squared_invalid_shape():
    """Test that invalid shapes raise ValueError."""
    x_1d = make_example_1d_numpy(10, random_state=1)
    y_2d = make_example_2d_numpy_series(10, 1, random_state=2)

    with pytest.raises(ValueError, match="x and y must be 1D or 2D"):
        squared_distance(x_1d, y_2d)


def test_squared_pairwise_self():
    """Test pairwise distance matrix to self."""
    X = make_example_3d_numpy(5, 1, 10, random_state=42, return_y=False)
    pw = squared_pairwise_distance(X)

    assert pw.shape == (5, 5)
    assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)
    assert np.allclose(pw, pw.T)
    assert np.all(pw >= 0)


def test_squared_pairwise_manual_comparison():
    """Test pairwise computation against manual calculation."""
    X = make_example_3d_numpy(4, 1, 8, random_state=1, return_y=False)
    pw = squared_pairwise_distance(X)

    manual_pw = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            manual_pw[i, j] = squared_distance(X[i], X[j])

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_squared_pairwise_multiple_to_multiple():
    """Test pairwise between two different collections."""
    X = make_example_3d_numpy(5, 1, 10, random_state=1, return_y=False)
    Y = make_example_3d_numpy(7, 1, 10, random_state=2, return_y=False)

    pw = squared_pairwise_distance(X, Y)
    assert pw.shape == (5, 7)
    assert np.all(pw >= 0)


def test_squared_pairwise_unequal_length():
    """Test pairwise with unequal length series."""
    X = make_example_3d_numpy_list(5, 1, random_state=1, return_y=False)
    pw = squared_pairwise_distance(X)

    assert pw.shape == (5, 5)
    assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_squared_pairwise_parallel(n_jobs):
    """Test parallel execution yields same result as serial."""
    X = make_example_3d_numpy(6, 1, 15, random_state=42, return_y=False)

    serial = squared_pairwise_distance(X, n_jobs=1)
    parallel = squared_pairwise_distance(X, n_jobs=n_jobs)

    assert_almost_equal(serial, parallel, decimal=10)


def test_squared_consistency_with_numpy():
    """Validate against numpy implementation."""
    x = make_example_1d_numpy(25, random_state=7)
    y = make_example_1d_numpy(25, random_state=13)

    result = squared_distance(x, y)
    expected = np.sum((x - y) ** 2)

    assert_almost_equal(result, expected, decimal=10)
