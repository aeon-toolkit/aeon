"""Tests for Minkowski distance functions.

This module tests minkowski_distance and minkowski_pairwise_distance,
validating correctness for different p values and weighted variants.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances.pointwise import (
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    minkowski_pairwise_distance,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.testing_config import MULTITHREAD_TESTING


def test_minkowski_p_equals_1_is_manhattan():
    """Test that Minkowski with p=1 equals Manhattan distance."""
    x = make_example_1d_numpy(20, random_state=1)
    y = make_example_1d_numpy(20, random_state=2)

    minkowski_p1 = minkowski_distance(x, y, p=1)
    manhattan_dist = manhattan_distance(x, y)

    assert_almost_equal(minkowski_p1, manhattan_dist, decimal=10)


def test_minkowski_p_equals_2_is_euclidean():
    """Test that Minkowski with p=2 equals Euclidean distance."""
    x = make_example_1d_numpy(20, random_state=1)
    y = make_example_1d_numpy(20, random_state=2)

    minkowski_p2 = minkowski_distance(x, y, p=2)
    euclidean_dist = euclidean_distance(x, y)

    assert_almost_equal(minkowski_p2, euclidean_dist, decimal=10)


def test_minkowski_basic_correctness():
    """Test basic correctness with known values."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # p=1: |1-4| + |2-5| + |3-6| = 9
    result_p1 = minkowski_distance(x, y, p=1.0)
    assert_almost_equal(result_p1, 9.0, decimal=10)

    # p=2: sqrt(9 + 9 + 9) = sqrt(27)
    result_p2 = minkowski_distance(x, y, p=2.0)
    assert_almost_equal(result_p2, np.sqrt(27.0), decimal=10)

    # p=3: (3^3 + 3^3 + 3^3)^(1/3) = (81)^(1/3)
    result_p3 = minkowski_distance(x, y, p=3.0)
    assert_almost_equal(result_p3, 81.0 ** (1.0 / 3.0), decimal=10)


def test_minkowski_identity():
    """Test that distance from series to itself is zero."""
    x = make_example_1d_numpy(15, random_state=42)

    for p in [1.0, 2.0, 3.0, 5.0, 10.0]:
        result = minkowski_distance(x, x, p=p)
        assert_almost_equal(result, 0.0, decimal=10)


def test_minkowski_symmetry():
    """Test that distance is symmetric."""
    x = make_example_1d_numpy(20, random_state=1)
    y = make_example_1d_numpy(20, random_state=2)

    for p in [1.0, 2.0, 3.5, 10.0]:
        d1 = minkowski_distance(x, y, p=p)
        d2 = minkowski_distance(y, x, p=p)
        assert_almost_equal(d1, d2, decimal=10)


def test_minkowski_non_negativity():
    """Test that distances are always non-negative."""
    x = make_example_1d_numpy(10, random_state=1)
    y = make_example_1d_numpy(10, random_state=2)

    for p in [1.0, 2.0, 3.0, 5.0]:
        result = minkowski_distance(x, y, p=p)
        assert result >= 0


def test_minkowski_parameter_validation():
    """Test that invalid parameters raise errors."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # p < 1 should raise ValueError
    with pytest.raises(ValueError, match="p should be greater or equal to 1"):
        minkowski_distance(x, y, p=0.5)

    with pytest.raises(ValueError, match="p should be greater or equal to 1"):
        minkowski_distance(x, y, p=0)


def test_minkowski_weighted():
    """Test Minkowski distance with custom weights."""
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    w = np.array([2.0, 2.0, 2.0])

    # With p=2: sqrt(2*1^2 + 2*1^2 + 2*0^2) = sqrt(4) = 2
    result = minkowski_distance(x, y, p=2.0, w=w)
    assert_almost_equal(result, 2.0, decimal=10)


def test_minkowski_weight_validation():
    """Test weight parameter validation."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # Mismatched weight shape
    w_wrong_shape = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Weights w must have the same shape"):
        minkowski_distance(x, y, p=2.0, w=w_wrong_shape)

    # Negative weights
    w_negative = np.array([1.0, -2.0, 3.0])
    with pytest.raises(ValueError, match="Input weights should be all non-negative"):
        minkowski_distance(x, y, p=2.0, w=w_negative)


def test_minkowski_multivariate():
    """Test with multivariate series."""
    x = make_example_2d_numpy_series(20, 5, random_state=1)
    y = make_example_2d_numpy_series(20, 5, random_state=2)

    for p in [1.0, 2.0, 3.0]:
        result = minkowski_distance(x, y, p=p)
        assert isinstance(result, float)
        assert result > 0


def test_minkowski_multivariate_weighted():
    """Test multivariate with weights."""
    x = make_example_2d_numpy_series(10, 3, random_state=1)
    w = np.ones_like(x) * 2.0
    y = make_example_2d_numpy_series(10, 3, random_state=2)

    result = minkowski_distance(x, y, p=2.0, w=w)
    assert isinstance(result, float)
    assert result > 0


def test_minkowski_different_channel_counts():
    """Test multivariate with different channel counts."""
    x = make_example_2d_numpy_series(10, 3, random_state=1)
    y = make_example_2d_numpy_series(10, 5, random_state=2)

    result = minkowski_distance(x, y, p=2.0)

    # Should use minimum channels
    y_truncated = y[:3, :]
    result_expected = minkowski_distance(x, y_truncated, p=2.0)
    assert_almost_equal(result, result_expected, decimal=10)


def test_minkowski_invalid_shape():
    """Test that invalid shapes raise ValueError."""
    x_1d = make_example_1d_numpy(10, random_state=1)
    y_2d = make_example_2d_numpy_series(10, 1, random_state=2)

    with pytest.raises(ValueError, match="Inconsistent dimensions"):
        minkowski_distance(x_1d, y_2d, p=2.0)


def test_minkowski_pairwise_self():
    """Test pairwise distance matrix to self."""
    X = make_example_3d_numpy(5, 1, 10, random_state=42, return_y=False)

    for p in [1.0, 2.0, 3.0]:
        pw = minkowski_pairwise_distance(X, p=p)

        assert pw.shape == (5, 5)
        assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)
        assert np.allclose(pw, pw.T)
        assert np.all(pw >= 0)


def test_minkowski_pairwise_manual_comparison():
    """Test pairwise computation against manual calculation."""
    X = make_example_3d_numpy(4, 1, 8, random_state=1, return_y=False)
    p = 2.5

    pw = minkowski_pairwise_distance(X, p=p)

    manual_pw = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            manual_pw[i, j] = minkowski_distance(X[i], X[j], p=p)

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_minkowski_pairwise_multiple_to_multiple():
    """Test pairwise between two different collections."""
    X = make_example_3d_numpy(5, 1, 10, random_state=1, return_y=False)
    Y = make_example_3d_numpy(7, 1, 10, random_state=2, return_y=False)

    pw = minkowski_pairwise_distance(X, Y, p=2.0)
    assert pw.shape == (5, 7)
    assert np.all(pw >= 0)


def test_minkowski_pairwise_weighted():
    """Test pairwise with weights."""
    X = make_example_2d_numpy_series(10, 4, random_state=1)
    X_collection = np.array([X, X, X])  # 3 instances
    w = np.ones_like(X) * 1.5

    pw = minkowski_pairwise_distance(X_collection, p=2.0, w=w)
    assert pw.shape == (3, 3)


def test_minkowski_pairwise_unequal_length():
    """Test pairwise with unequal length series."""
    X = make_example_3d_numpy_list(5, 1, random_state=1, return_y=False)
    pw = minkowski_pairwise_distance(X, p=2.0)

    assert pw.shape == (5, 5)
    assert_almost_equal(np.diag(pw), np.zeros(5), decimal=10)


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_minkowski_pairwise_parallel(n_jobs):
    """Test parallel execution yields same result as serial."""
    X = make_example_3d_numpy(6, 1, 15, random_state=42, return_y=False)

    serial = minkowski_pairwise_distance(X, p=2.0, n_jobs=1)
    parallel = minkowski_pairwise_distance(X, p=2.0, n_jobs=n_jobs)

    assert_almost_equal(serial, parallel, decimal=10)


def test_minkowski_large_p():
    """Test with large p values (approaching infinity norm)."""
    x = np.array([1.0, 2.0, 10.0, 4.0])
    y = np.array([2.0, 3.0, 15.0, 5.0])

    # As p approaches infinity, Minkowski approaches Chebyshev (max difference)
    # max difference here is |10-15| = 5
    result_large_p = minkowski_distance(x, y, p=100.0)

    # Should be close to max absolute difference
    max_diff = np.max(np.abs(x - y))
    assert result_large_p > max_diff * 0.9  # Within 10% of max
    assert result_large_p <= max_diff * 1.1


def test_minkowski_consistency_with_scipy():
    """Validate against scipy implementation."""
    try:
        from scipy.spatial.distance import minkowski as scipy_minkowski

        x = make_example_1d_numpy(25, random_state=7)
        y = make_example_1d_numpy(25, random_state=13)

        for p in [1.0, 2.0, 3.0, 5.0]:
            result = minkowski_distance(x, y, p=p)
            expected = scipy_minkowski(x, y, p=p)
            assert_almost_equal(result, expected, decimal=10)
    except ImportError:
        pytest.skip("scipy not available")
