"""Tests for TopKSimilaritySearch."""

__author__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon.similarity_search.top_k_similarity import TopKSimilaritySearch

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_TopKSimilaritySearch(dtype):
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = TopKSimilaritySearch(k=1)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = TopKSimilaritySearch(k=3)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    search = TopKSimilaritySearch(k=1, normalize=True)
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])

    search = TopKSimilaritySearch(k=1, normalize=True)
    search.fit(X)
    idx = search.predict(q, q_index=(1, 2))
    assert_array_equal(idx, [(1, 0)])
