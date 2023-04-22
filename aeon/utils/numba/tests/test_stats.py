# -*- coding: utf-8 -*-
"""Tests for RDST numba utils functions."""

__author__ = ["baraline"]

import numpy as np
from numpy.testing import assert_array_equal

from aeon.utils.numba.stats import is_prime, prime_up_to


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
