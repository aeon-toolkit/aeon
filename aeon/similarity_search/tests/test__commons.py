"""Test _commons.py functions."""

__maintainer__ = ["baraline"]

import numpy as np
from numpy.testing import assert_array_almost_equal

from aeon.similarity_search._commons import fft_sliding_dot_product


def test_fft_sliding_dot_product():
    """Test the fft_sliding_dot_product function."""
    X = np.random.rand(1, 10)
    q = np.random.rand(1, 5)

    values = fft_sliding_dot_product(X, q)

    assert_array_almost_equal(
        values[0],
        [np.dot(q[0], X[0, i : i + 5]) for i in range(X.shape[1] - 5 + 1)],
    )
