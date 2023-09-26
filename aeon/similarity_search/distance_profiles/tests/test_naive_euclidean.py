# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:21:00 2023

@author: antoi
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.distances import euclidean_distance
from aeon.similarity_search.distance_profiles.naive_euclidean import (
    naive_euclidean_profile,
)

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_naive_euclidean(dtype):
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    Q = np.asarray([[3, 4, 5]], dtype=dtype)

    dist_profile = naive_euclidean_profile(X, Q)

    expected = np.array(
        [
            [
                euclidean_distance(Q, X[j, :, i : i + Q.shape[-1]])
                for i in range(X.shape[-1] - Q.shape[-1] + 1)
            ]
            for j in range(X.shape[0])
        ]
    )
    assert_array_almost_equal(dist_profile, expected)
