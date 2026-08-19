"""Tests for Gauss transformation."""

import numpy as np
import pytest

from aeon.transformations.series.smoothing import GaussianFilter


@pytest.mark.parametrize("sigma", [0.1, 1, 10])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_gauss(sigma, order):
    """Test the functionality of Gauss transformation."""
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

    sg = GaussianFilter(sigma=sigma, order=order)
    x_1 = sg.fit_transform(x1)
    x_2 = sg.fit_transform(x2)
    x_12 = sg.fit_transform(x12)

    np.testing.assert_almost_equal(x_1[0], x_12[0], decimal=4)
    np.testing.assert_almost_equal(x_2[0], x_12[1], decimal=4)
