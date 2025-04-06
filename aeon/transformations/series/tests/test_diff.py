"""Tests for Difference transformation."""

import numpy as np

from aeon.transformations.series._diff import DifferenceTransformer


def test_diff():
    """Tests basic first and second order differencing."""
    X = np.array([[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]])

    dt1 = DifferenceTransformer(order=1)
    Xt1 = dt1.fit_transform(X)
    expected1 = np.array([[np.nan, 3.0, 5.0, 7.0, 9.0, 11.0]])

    assert Xt1.shape == X.shape, "Shape mismatch for order 1"
    np.testing.assert_allclose(
        Xt1, expected1, equal_nan=True, err_msg="Value mismatch for order 1"
    )

    dt2 = DifferenceTransformer(order=2)
    Xt2 = dt2.fit_transform(X)
    expected2 = np.array([[np.nan, np.nan, 2.0, 2.0, 2.0, 2.0]])

    assert Xt2.shape == X.shape, "Shape mismatch for order 2"
    np.testing.assert_allclose(
        Xt2, expected2, equal_nan=True, err_msg="Value mismatch for order 2"
    )
