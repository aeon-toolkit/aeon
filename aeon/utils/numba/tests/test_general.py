"""Tests for numba functions."""

__maintainer__ = []

import numpy as np
import pytest
from numba import njit
from numba.core.registry import CPUDispatcher
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.utils.numba.general import (
    combinations_1d,
    generate_new_default_njit_func,
    get_subsequence,
    get_subsequence_with_mean_std,
    sliding_dot_product,
    sliding_mean_std_one_series,
    z_normalise_series,
    z_normalize_series_with_mean_std,
)

DATATYPES = ["int32", "int64", "float32", "float64"]


def test_generate_new_default_njit_func():
    """Test the generation of a new njit function with modified default arguments."""

    def _dummy_func(x, arg1=0.0, arg2=1.0):
        return x - arg1 + arg2

    dummy_func = njit(_dummy_func, fastmath=True)

    new_dummy_func = generate_new_default_njit_func(dummy_func, {"arg1": -1.0})

    expected_targetoptions = {"fastmath": True, "nopython": True, "boundscheck": None}

    if isinstance(dummy_func, CPUDispatcher):
        assert dummy_func.py_func.__defaults__ == (0.0, 1.0)
        assert new_dummy_func.py_func.__defaults__ == (-1.0, 1.0)

        assert dummy_func.targetoptions == expected_targetoptions
        assert new_dummy_func.targetoptions == expected_targetoptions

        assert dummy_func.__name__ != new_dummy_func.__name__
        assert dummy_func.py_func.__code__ == new_dummy_func.py_func.__code__

    elif callable(dummy_func):
        assert dummy_func.__defaults__ == (0.0, 1.0)
        assert new_dummy_func.__defaults__ == (-1.0, 1.0)
        assert dummy_func.__name__ != new_dummy_func.__name__
        assert dummy_func.__code__ == new_dummy_func.__code__


@pytest.mark.parametrize("type", DATATYPES)
def test_z_normalize_series_with_mean_std(type):
    """Test z-normalization of a series using mean and standard deviation."""
    a = np.array([2, 2, 2], dtype=type)
    a_expected = np.array([0, 0, 0], dtype=type)
    a_result = z_normalize_series_with_mean_std(a, a.mean(), a.std())
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
def test_sliding_dot_product(dtype):
    """Test sliding dot product computation."""
    X = np.random.rand(3, 150).astype(dtype)
    for length in [3, 5, 11]:
        for dilation in [1, 3, 5, 6]:
            values = np.random.rand(3, length).astype(dtype)
            dots = sliding_dot_product(X, values, length, dilation)
            for i_sub in range(X.shape[1] - (length - 1) * dilation):
                _idx = [i_sub + j * dilation for j in range(length)]
                assert_array_almost_equal(
                    (X[:, _idx] * values).sum(axis=1), dots[:, i_sub]
                )


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
