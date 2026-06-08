"""Tests for Euclidean distance functions.

This module contains comprehensive tests for euclidean_distance and
euclidean_pairwise_distance, including correctness validation, mathematical
properties, edge cases, and pairwise matrix computation.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances.pointwise import euclidean_distance, euclidean_pairwise_distance
from aeon.distances.pointwise._squared import squared_distance
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_collection,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.testing_config import MULTITHREAD_TESTING


def test_euclidean_basic_correctness():
    """Test basic correctness with known values."""
    # Simple 1D case
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    # sqrt((3^2 + 3^2 + 3^2)) = sqrt(27) = 5.196...
    expected = np.sqrt(27.0)
    result = euclidean_distance(x, y)
    assert_almost_equal(result, expected, decimal=10)

    # 2D univariate case
    x_2d = np.array([[1.0, 2.0, 3.0]])
    y_2d = np.array([[4.0, 5.0, 6.0]])
    result_2d = euclidean_distance(x_2d, y_2d)
    assert_almost_equal(result_2d, expected, decimal=10)


def test_euclidean_relationship_to_squared():
    """Test that euclidean equals sqrt of squared distance."""
    x = make_example_1d_numpy(20, random_state=42)
    y = make_example_1d_numpy(20, random_state=24)

    euclidean_result = euclidean_distance(x, y)
    squared_result = squared_distance(x, y)

    assert_almost_equal(euclidean_result, np.sqrt(squared_result), decimal=10)
    assert_almost_equal(euclidean_result**2, squared_result, decimal=10)


def test_euclidean_identity():
    """Test that distance from series to itself is zero."""
    # 1D case
    x = make_example_1d_numpy(15, random_state=42)
    result = euclidean_distance(x, x)
    assert_almost_equal(result, 0.0, decimal=10)

    # 2D univariate case
    x_2d = make_example_2d_numpy_series(15, 1, random_state=42)
    result_2d = euclidean_distance(x_2d, x_2d)
    assert_almost_equal(result_2d, 0.0, decimal=10)

    # 2D multivariate case
    x_multi = make_example_2d_numpy_series(15, 5, random_state=42)
    result_multi = euclidean_distance(x_multi, x_multi)
    assert_almost_equal(result_multi, 0.0, decimal=10)


def test_euclidean_symmetry():
    """Test that distance is symmetric."""
    x = make_example_1d_numpy(20, random_state=1)
    y = make_example_1d_numpy(20, random_state=2)

    d1 = euclidean_distance(x, y)
    d2 = euclidean_distance(y, x)

    assert_almost_equal(d1, d2, decimal=10)


def test_euclidean_non_negativity():
    """Test that distances are always non-negative."""
    # Test with multiple random seeds
    for seed in [1, 10, 42, 99, 123]:
        x = make_example_1d_numpy(10, random_state=seed)
        y = make_example_1d_numpy(10, random_state=seed + 100)
        result = euclidean_distance(x, y)
        assert result >= 0, f"Distance should be non-negative, got {result}"


def test_euclidean_triangle_inequality():
    """Test triangle inequality: d(x,z) <= d(x,y) + d(y,z)."""
    x = make_example_1d_numpy(15, random_state=1)
    y = make_example_1d_numpy(15, random_state=2)
    z = make_example_1d_numpy(15, random_state=3)

    d_xz = euclidean_distance(x, z)
    d_xy = euclidean_distance(x, y)
    d_yz = euclidean_distance(y, z)

    # Allow small numerical tolerance
    assert d_xz <= d_xy + d_yz + 1e-10


def test_euclidean_shape_1d():
    """Test with 1D arrays (n_timepoints,)."""
    x = make_example_1d_numpy(20, random_state=1)
    y = make_example_1d_numpy(20, random_state=2)

    result = euclidean_distance(x, y)
    assert isinstance(result, float)
    assert result > 0


def test_euclidean_shape_2d_univariate():
    """Test with 2D univariate arrays (1, n_timepoints)."""
    x = make_example_2d_numpy_series(20, 1, random_state=1)
    y = make_example_2d_numpy_series(20, 1, random_state=2)

    result = euclidean_distance(x, y)
    assert isinstance(result, float)
    assert result > 0


def test_euclidean_shape_2d_multivariate():
    """Test with 2D multivariate arrays (n_channels, n_timepoints)."""
    x = make_example_2d_numpy_series(20, 5, random_state=1)
    y = make_example_2d_numpy_series(20, 5, random_state=2)

    result = euclidean_distance(x, y)
    assert isinstance(result, float)
    assert result > 0


def test_euclidean_single_point():
    """Test with single point series."""
    # 1D single point
    x = np.array([5.0])
    y = np.array([10.0])
    result = euclidean_distance(x, y)
    assert_almost_equal(result, 5.0, decimal=10)

    # 2D single point
    x_2d = np.array([[5.0]])
    y_2d = np.array([[10.0]])
    result_2d = euclidean_distance(x_2d, y_2d)
    assert_almost_equal(result_2d, 5.0, decimal=10)


def test_euclidean_equal_series():
    """Test that equal series have zero distance."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x.copy()

    result = euclidean_distance(x, y)
    assert_almost_equal(result, 0.0, decimal=10)


def test_euclidean_extreme_values():
    """Test with extreme but valid floating point values."""
    # Large values
    x_large = np.array([1e100, 2e100, 3e100])
    y_large = np.array([1e100, 2e100, 3e100])
    result_large = euclidean_distance(x_large, y_large)
    assert_almost_equal(result_large, 0.0, decimal=5)

    # Small values
    x_small = np.array([1e-100, 2e-100, 3e-100])
    y_small = np.array([4e-100, 5e-100, 6e-100])
    result_small = euclidean_distance(x_small, y_small)
    assert result_small >= 0
    assert result_small < 1e-99


def test_euclidean_different_channel_counts():
    """Test that multivariate distance uses minimum channel count."""
    # x has 3 channels, y has 5 channels - should use first 3
    x = make_example_2d_numpy_series(10, 3, random_state=1)
    y = make_example_2d_numpy_series(10, 5, random_state=2)

    result = euclidean_distance(x, y)
    assert isinstance(result, float)
    assert result >= 0

    # Should be same as if we manually take first 3 channels of y
    y_truncated = y[:3, :]
    result_expected = euclidean_distance(x, y_truncated)
    assert_almost_equal(result, result_expected, decimal=10)


def test_euclidean_invalid_shape():
    """Test that invalid shapes raise ValueError."""
    x_1d = make_example_1d_numpy(10, random_state=1)
    y_2d = make_example_2d_numpy_series(10, 1, random_state=2)

    with pytest.raises(ValueError, match="x and y must be 1D or 2D"):
        euclidean_distance(x_1d, y_2d)


def test_euclidean_pairwise_self():
    """Test pairwise distance matrix to self."""
    X = make_example_3d_numpy(5, 1, 10, random_state=42, return_y=False)
    pw = euclidean_pairwise_distance(X)

    # Check shape
    assert pw.shape == (5, 5)

    # Check diagonal is zero (distance to self)
    diag = np.diag(pw)
    assert_almost_equal(diag, np.zeros(5), decimal=10)

    # Check symmetry
    assert np.allclose(pw, pw.T)

    # Check all non-negative
    assert np.all(pw >= 0)


def test_euclidean_pairwise_manual_comparison():
    """Test pairwise computation against manual calculation."""
    X = make_example_3d_numpy(4, 1, 8, random_state=1, return_y=False)
    pw = euclidean_pairwise_distance(X)

    # Manually compute distances
    manual_pw = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            manual_pw[i, j] = euclidean_distance(X[i], X[j])

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_euclidean_pairwise_multiple_to_multiple():
    """Test pairwise distance between two different collections."""
    X = make_example_3d_numpy(5, 1, 10, random_state=1, return_y=False)
    Y = make_example_3d_numpy(7, 1, 10, random_state=2, return_y=False)

    pw = euclidean_pairwise_distance(X, Y)

    # Check shape
    assert pw.shape == (5, 7)

    # Check all non-negative
    assert np.all(pw >= 0)

    # Manually verify a few entries
    assert_almost_equal(pw[0, 0], euclidean_distance(X[0], Y[0]), decimal=10)
    assert_almost_equal(pw[2, 3], euclidean_distance(X[2], Y[3]), decimal=10)


def test_euclidean_pairwise_multivariate():
    """Test pairwise distance with multivariate series."""
    X = make_example_3d_numpy(4, 5, 12, random_state=42, return_y=False)
    pw = euclidean_pairwise_distance(X)

    assert pw.shape == (4, 4)
    assert_almost_equal(np.diag(pw), np.zeros(4), decimal=10)
    assert np.allclose(pw, pw.T)


def test_euclidean_pairwise_unequal_length():
    """Test pairwise distance with unequal length series."""
    X = make_example_3d_numpy_list(5, 1, random_state=1, return_y=False)
    pw = euclidean_pairwise_distance(X)

    assert pw.shape == (5, 5)
    assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)
    assert np.allclose(pw, pw.T)


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_euclidean_pairwise_parallel(n_jobs):
    """Test that parallel execution yields same result as serial."""
    X = make_example_3d_numpy(6, 1, 15, random_state=42, return_y=False)

    serial = euclidean_pairwise_distance(X, n_jobs=1)
    parallel = euclidean_pairwise_distance(X, n_jobs=n_jobs)

    assert isinstance(parallel, np.ndarray)
    assert serial.shape == parallel.shape
    assert_almost_equal(serial, parallel, decimal=10)


def test_euclidean_consistency_with_numpy():
    """Validate against numpy.linalg.norm implementation."""
    x = make_example_1d_numpy(25, random_state=7)
    y = make_example_1d_numpy(25, random_state=13)

    result = euclidean_distance(x, y)
    expected = np.linalg.norm(x - y)

    assert_almost_equal(result, expected, decimal=10)


def test_euclidean_pairwise_2d_collection():
    """Test pairwise with 2D collection (n_cases, n_timepoints)."""
    X = make_example_2d_numpy_collection(5, 10, random_state=1, return_y=False)
    pw = euclidean_pairwise_distance(X)

    assert pw.shape == (5, 5)
    assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)
