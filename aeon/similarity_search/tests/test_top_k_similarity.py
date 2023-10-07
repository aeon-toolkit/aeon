"""
Created on Sat Sep  9 14:12:58 2023

@author: antoi
"""


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
    Q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = TopKSimilaritySearch(k=1)
    search.fit(X)
    idx = search.predict(Q)
    assert_array_equal(idx, [(0, 2)])

    search = TopKSimilaritySearch(k=3)
    search.fit(X)
    idx = search.predict(Q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])
