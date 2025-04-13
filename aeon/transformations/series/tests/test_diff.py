"""Tests for Difference transformation."""

import numpy as np

from aeon.transformations.series._diff import DifferenceTransformer


def test_diff():
    """Tests basic first and second order differencing."""
    X = np.array([[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]])

    dt1 = DifferenceTransformer(order=1)
    Xt1 = dt1.fit_transform(X)
    expected1 = np.array([[3.0, 5.0, 7.0, 9.0, 11.0]])

    np.testing.assert_allclose(
        Xt1, expected1, equal_nan=True, err_msg="Value mismatch for order 1"
    )

    dt2 = DifferenceTransformer(order=2)
    Xt2 = dt2.fit_transform(X)
    expected2 = np.array([[2.0, 2.0, 2.0, 2.0]])

    np.testing.assert_allclose(
        Xt2, expected2, equal_nan=True, err_msg="Value mismatch for order 2"
    )

    Y = np.array([[1, 2, 3, 4], [5, 3, 1, 8]])

    Yt1 = dt1.fit_transform(Y)
    expected3 = np.array([[1, 1, 1], [-2, -2, 7]])
    np.testing.assert_allclose(
        Yt1,
        expected3,
        equal_nan=True,
        err_msg="Value mismatch for order 1,multivariate",
    )
