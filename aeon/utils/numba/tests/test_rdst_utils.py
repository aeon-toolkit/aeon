# -*- coding: utf-8 -*-
"""Tests for RDST numba utils functions."""

__author__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.utils.numba.rdst_utils import (
    get_subsequence,
    get_subsequence_with_mean_std,
    is_prime,
    prime_up_to,
)

DATATYPES = ["int32", "int64", "float32", "float64"]


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


@pytest.mark.parametrize("type", DATATYPES)
def test_get_subsequence(type):
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=type)
    # get_subsequence(i_start, length, dilation)
    sub = get_subsequence(x, 2, 3, 1)
    assert_array_equal(x[:, [2, 3, 4]], sub)
    sub = get_subsequence(x, 2, 3, 3)
    assert_array_equal(x[:, [2, 5, 8]], sub)


@pytest.mark.parametrize("type", DATATYPES)
def test_get_subsequence_with_mean_std(type):
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=type)
    # i_start, length, dilation
    sub, mean, std = get_subsequence_with_mean_std(x, 2, 3, 1)
    assert_array_equal(x[:, [2, 3, 4]], sub)
    assert_array_almost_equal(mean, sub.mean(axis=1))
    assert_array_almost_equal(std, sub.std(axis=1))

    sub, mean, std = get_subsequence_with_mean_std(x, 2, 3, 3)
    assert_array_equal(x[:, [2, 5, 8]], sub)
    assert_array_almost_equal(mean, sub.mean(axis=1))
    assert_array_almost_equal(std, sub.std(axis=1))


@pytest.mark.parametrize("type", DATATYPES)
def test_sliding_mean_std_one_series(type):
    pass


# sliding_mean_std_one_series(X, length, dilation)

# sliding_dot_product(X, values, length, dilation)

# compute_normalized_shapelet_dist_vector(X, values, length, dilation, means, stds)


def test_compute_normalized_shapelet_dist_vector():
    # (X, i_start, length, dilation)
    pass


def test_compute_shapelet_dist_vector():
    pass
