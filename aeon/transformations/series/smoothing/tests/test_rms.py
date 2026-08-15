"""Tests for SIV transformation."""

import numpy as np
import pytest

from aeon.transformations.series.smoothing import RecursiveMedianSieve


@pytest.mark.parametrize(
    "window_length", [1, 2, 3, 5, 7, 10, 11, [2, 3], [3, 5], [3, 5, 7], [3, 5, 7, 11]]
)
def test_siv(window_length):
    """Test the functionality of SIV transformation."""
    n_samples = 100
    t = np.linspace(0, 10, n_samples)
    x1 = (
        0.5 * np.sin(2 * np.pi * 1 * t)
        + 0.2 * np.sin(2 * np.pi * 5 * t)
        + 0.1 * np.sin(2 * np.pi * 10 * t)
    )
    x2 = (
        0.4 * np.sin(2 * np.pi * 1.5 * t)
        + 0.3 * np.sin(2 * np.pi * 4 * t)
        + 0.1 * np.sin(2 * np.pi * 8 * t)
    )
    x12 = np.array([x1, x2])

    siv = RecursiveMedianSieve(window_length=window_length)
    x_1 = siv.fit_transform(x1)
    x_2 = siv.fit_transform(x2)
    x_12 = siv.fit_transform(x12)

    np.testing.assert_almost_equal(x_1[0], x_12[0], decimal=4)
    np.testing.assert_almost_equal(x_2[0], x_12[1], decimal=4)


def test_siv_default_window_length():
    """Test ``window_length=None`` uses the documented [3, 5, 7] sieve windows."""
    x = np.array([[1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0, 0.0]])

    siv_default = RecursiveMedianSieve()
    siv_explicit = RecursiveMedianSieve(window_length=[3, 5, 7])

    np.testing.assert_almost_equal(
        siv_default.fit_transform(x), siv_explicit.fit_transform(x)
    )
