# -*- coding: utf-8 -*-
"""Tests for RDST numba utils functions."""

__author__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.utils.numba.rdst_utils import (
    combinations_1d,
    compute_normalized_shapelet_dist_vector,
    compute_shapelet_dist_vector,
    get_subsequence,
    get_subsequence_with_mean_std,
    is_prime,
    prime_up_to,
    sliding_dot_product,
    sliding_mean_std_one_series,
)

DATATYPES = ("int32", "int64", "float32", "float64")


def test_prime_up_to():
    true_primes_to_100 = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    primes = prime_up_to(100)
    assert_array_equal(true_primes_to_100, primes)


def test_is_prime():
    true_primes_to_100 = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    for n in np.arange(100):
        if n in true_primes_to_100:
            assert is_prime(n)
        else:
            assert not is_prime(n)


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
    X = np.random.rand(3, 50).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            mean, std = sliding_mean_std_one_series(X, length, dilation)
            for i_sub in range(X.shape[1] - (length - 1) * dilation):
                _idx = [i_sub + j * dilation for j in range(length)]
                assert_array_almost_equal(X[:, _idx].mean(axis=1), mean[:, i_sub])
                assert_array_almost_equal(X[:, _idx].std(axis=1), std[:, i_sub])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_sliding_dot_product(dtype):
    X = np.random.rand(3, 50).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            values = np.random.rand(3, length).astype(dtype)
            dots = sliding_dot_product(X, values, length, dilation)
            for i_sub in range(X.shape[1] - (length - 1) * dilation):
                _idx = [i_sub + j * dilation for j in range(length)]
                assert_array_almost_equal(
                    (X[:, _idx] * values).sum(axis=1), dots[:, i_sub]
                )


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_normalized_shapelet_dist_vector(dtype):
    # Constant case is tested with dtype int
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            X = (np.random.rand(3, 50)).astype(dtype)
            values = np.random.rand(3, length).astype(dtype)
            d_vect = compute_normalized_shapelet_dist_vector(
                X, values, length, dilation, values.mean(axis=1), values.std(axis=1)
            )
            norm_values = values - values.mean(axis=1, keepdims=True)
            for i_channel in range(X.shape[0]):
                if values[i_channel].std() > 0:
                    norm_values[i_channel] /= values[i_channel].std()
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                for i_channel in range(X.shape[0]):
                    norm_sub = X[i_channel, _idx]
                    norm_sub = norm_sub - norm_sub.mean()
                    if norm_sub.std() > 0:
                        norm_sub /= norm_sub.std()
                    true_vect[i_sub] += ((norm_values[i_channel] - norm_sub) ** 2).sum()
            if dtype == "float32":
                # Fastmath with float32 can sometime produce different of approx 0.0005
                # Any way to compensate this ?
                assert_array_almost_equal(d_vect, true_vect, decimal=3)
            else:
                assert_array_almost_equal(d_vect, true_vect)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_dist_vector(dtype):
    X = np.random.rand(3, 50).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            values = np.random.rand(3, length).astype(dtype)
            d_vect = compute_shapelet_dist_vector(X, values, length, dilation)
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                for i_channel in range(X.shape[0]):
                    _sub = X[i_channel, _idx]
                    true_vect[i_sub] += ((values[i_channel] - _sub) ** 2).sum()
            assert_array_almost_equal(d_vect, true_vect)


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
