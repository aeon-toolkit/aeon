"""Tests for DFT transformation."""

__maintainer__ = []

import numpy as np
import pytest


@pytest.mark.parametrize("r", [0.00, 0.25, 0.50, 0.75, 1.00])
@pytest.mark.parametrize("sort", [True, False])
def test_dft(r, sort):
    """Test the functionality of DFT transformation."""
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
    x12r = x12 + np.random.random((2, n_samples)) * 0.25

    from aeon.transformations.series._dft import DFTSeriesTransformer

    dft = DFTSeriesTransformer(r=r, sort=sort)
    x_1 = dft.fit_transform(x1)
    x_2 = dft.fit_transform(x2)
    x_12 = dft.fit_transform(x12)
    dft.fit_transform(x12r)

    np.testing.assert_almost_equal(x_1[0], x_12[0], decimal=4)
    np.testing.assert_almost_equal(x_2[0], x_12[1], decimal=4)
