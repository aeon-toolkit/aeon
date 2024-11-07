"""Tests for numba functions."""

__maintainer__ = []

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.utils.numba.general import (
    combinations_1d,
    get_all_subsequences,
    get_subsequence,
    get_subsequence_with_mean_std,
    normalise_subsequences,
    sliding_mean_std_one_series,
    z_normalise_series,
    z_normalise_series_with_mean_std,
)

DATATYPES = ["int32", "int64", "float32", "float64"]


@pytest.mark.parametrize("type", DATATYPES)
def test_z_normalise_series_with_mean_std(type):
    """Test z-normalization of a series using mean and standard deviation."""
    a = np.array([2, 2, 2], dtype=type)
    a_expected = np.array([0, 0, 0], dtype=type)
    a_result = z_normalise_series_with_mean_std(a, a.mean(), a.std())
    assert_array_equal(a_result, a_expected)


@pytest.mark.parametrize("type", DATATYPES)
def test_z_normalise_series(type):
    """Test the function z_normalise_series."""
    a = np.array([2, 2, 2], dtype=type)
    a_expected = np.array([0, 0, 0], dtype=type)
    a_result = z_normalise_series(a)
    assert_array_equal(a_result, a_expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_subsequence(dtype):
    """Test the extraction of subsequences from a 1D array."""
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=dtype)
    # get_subsequence(i_start, length, dilation)
    sub = get_subsequence(x, 2, 3, 1)
    assert_array_equal(x[:, [2, 3, 4]], sub)
    sub = get_subsequence(x, 2, 3, 3)
    assert_array_equal(x[:, [2, 5, 8]], sub)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_subsequence_with_mean_std(dtype):
    """Test the extraction of subsequences with mean and std from a 1D array."""
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=dtype)
    # i_start, length, dilation
    sub, mean, std = get_subsequence_with_mean_std(x, 2, 3, 1)
    assert_array_equal(x[:, [2, 3, 4]], sub)
    assert_array_almost_equal(mean, sub.mean(axis=1))
    assert_array_almost_equal(std, sub.std(axis=1))

    sub, mean, std = get_subsequence_with_mean_std(x, 2, 3, 3)
    assert_array_equal(x[:, [2, 5, 8]], sub)
    assert_array_almost_equal(mean, sub.mean(axis=1))
    assert_array_almost_equal(std, sub.std(axis=1))


@pytest.mark.parametrize("dtype", DATATYPES)
def test_sliding_mean_std_one_series(dtype):
    """Test sliding mean and standard deviation computations on a series."""
    X = np.random.rand(3, 150).astype(dtype)
    for length in [5, 50]:
        for dilation in [1, 3]:
            mean, std = sliding_mean_std_one_series(X, length, dilation)
            for i_sub in range(X.shape[1] - (length - 1) * dilation):
                _idx = [i_sub + j * dilation for j in range(length)]
                if dtype == "float32":
                    assert_array_almost_equal(
                        X[:, _idx].mean(axis=1), mean[:, i_sub], decimal=4
                    )
                    assert_array_almost_equal(
                        X[:, _idx].std(axis=1), std[:, i_sub], decimal=4
                    )
                else:
                    assert_array_almost_equal(X[:, _idx].mean(axis=1), mean[:, i_sub])
                    assert_array_almost_equal(X[:, _idx].std(axis=1), std[:, i_sub])

    # Test error on wrong dimension
    error_str = "Invalid input parameter for sliding mean and std computations"
    with pytest.raises(ValueError, match=error_str):
        mean, std = sliding_mean_std_one_series(X, 100, 3)

    with pytest.raises(ValueError, match=error_str):
        mean, std = sliding_mean_std_one_series(X, 100, 3)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_combinations_1d(dtype):
    """Test combinations of elements from two 1D arrays."""
    x = np.array([1, 1, 2, 2, 3, 3, 9, 4, 7, 9, 9], dtype=dtype)
    y = np.array([1, 1, 3, 5, 1, 3, 9, 2, 9, 9, 7], dtype=dtype)
    combs = combinations_1d(x, y)
    true_combs = np.array(
        [[1, 1], [2, 3], [2, 5], [3, 1], [3, 3], [9, 9], [4, 2], [7, 9], [9, 7]],
        dtype=dtype,
    )
    assert_array_equal(combs, true_combs)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalise_subsequences(dtype):
    """Test 3d z-normalization."""
    X = np.asarray([[[1, 1, 1]], [[1, 1, 1]]], dtype=dtype)
    # Transpose as this function expect means and std in (n channels, n_subsequence)
    X_norm = normalise_subsequences(X, X.mean(axis=2).T, X.std(axis=2).T)
    assert np.all(X_norm == 0)
    assert np.all(X.shape == X_norm.shape)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_all_subsequences(dtype):
    """Test generation of all subsequences."""
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype)
    length = 3
    dilation = 1
    X_subs = get_all_subsequences(X, length, dilation)
    X_true = np.asarray(
        [
            [[1, 2, 3]],
            [[2, 3, 4]],
            [[3, 4, 5]],
            [[4, 5, 6]],
            [[5, 6, 7]],
            [[6, 7, 8]],
        ],
        dtype=dtype,
    )
    assert_array_equal(X_subs, X_true)

    length = 3
    dilation = 2
    X_subs = get_all_subsequences(X, length, dilation)
    X_true = np.asarray(
        [
            [[1, 3, 5]],
            [[2, 4, 6]],
            [[3, 5, 7]],
            [[4, 6, 8]],
        ],
        dtype=dtype,
    )
    assert_array_equal(X_subs, X_true)
