"""Tests for STAMP algorithm."""

import numpy as np
from numpy.testing import assert_array_almost_equal

from aeon.similarity_search._commons import get_ith_products
from aeon.similarity_search.matrix_profiles.stamp import _update_dot_product


def test_update_dot_product():
    """Test the _update_dot_product function."""
    X = np.random.rand(1, 50)
    T = np.random.rand(1, 25)
    L = 10
    current_product = get_ith_products(X, T, L, 0)
    for i_query in range(1, T.shape[1] - L + 1):
        new_product = get_ith_products(
            X,
            T,
            L,
            i_query,
        )
        current_product = _update_dot_product(
            X,
            T,
            current_product,
            L,
            i_query,
        )
        assert_array_almost_equal(new_product, current_product)
