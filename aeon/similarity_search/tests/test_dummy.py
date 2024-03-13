"""Tests for DummySimilaritySearch."""

__maintainer__ = []


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon.similarity_search._dummy import DummySimilaritySearch

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_DummySimilaritySearch(dtype):
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=dtype)
    q = np.asarray([[3, 4, 5, 6]], dtype=dtype)

    search = DummySimilaritySearch()
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, 2)

    search = DummySimilaritySearch(normalize=True)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, 0)

    search = DummySimilaritySearch(normalize=True)
    search.fit(X)
    idx = search.predict(q, q_index=0)
    assert_array_equal(idx, 2)
