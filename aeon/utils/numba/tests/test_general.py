# -*- coding: utf-8 -*-
"""Tests for numba functions."""

__author__ = ["TonyBagnall", "baraline"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.utils.numba.general import (
    combinations_1d,
    get_subsequence,
    get_subsequence_with_mean_std,
    sliding_dot_product,
    sliding_mean_std_one_series,
    z_normalise_series,
)

DATATYPES = ["int32", "int64", "float32", "float64"]


@pytest.mark.parametrize("type", DATATYPES)
def test_z_normalise_series(type):
    """Test the function z_normalise_series."""
    a = np.array([2, 2, 2], dtype=type)
    a_expected = np.array([0, 0, 0], dtype=type)
    a_result = z_normalise_series(a)
    assert_array_equal(a_result, a_expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_subsequence(dtype):
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=dtype)
    # get_subsequence(i_start, length, dilation)
    sub = get_subsequence(x, 2, 3, 1)
    assert_array_equal(x[:, [2, 3, 4]], sub)
    sub = get_subsequence(x, 2, 3, 3)
    assert_array_equal(x[:, [2, 5, 8]], sub)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_subsequence_with_mean_std(dtype):
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
    X = np.random.rand(3, 150).astype(dtype)
    for length in [3, 5, 11]:
        for dilation in [1, 3, 5, 6]:
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


@pytest.mark.parametrize("dtype", DATATYPES)
def test_sliding_dot_product(dtype):
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
    x = np.array([1, 1, 2, 2, 3, 3, 9, 4, 7, 9, 9], dtype=dtype)
    y = np.array([1, 1, 3, 5, 1, 3, 9, 2, 9, 9, 7], dtype=dtype)
    combs = combinations_1d(x, y)
    true_combs = np.array(
        [[1, 1], [2, 3], [2, 5], [3, 1], [3, 3], [9, 9], [4, 2], [7, 9], [9, 7]],
        dtype=dtype,
    )
    assert_array_equal(combs, true_combs)
