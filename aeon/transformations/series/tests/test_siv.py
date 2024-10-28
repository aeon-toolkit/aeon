"""Tests for SIV transformation."""

__maintainer__ = []

import numpy as np
import pytest


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
    x12r = x12 + np.random.random((2, n_samples)) * 0.25

    from aeon.transformations.series._siv import SIVSeriesTransformer

    siv = SIVSeriesTransformer(window_length=window_length)
    x_1 = siv.fit_transform(x1)
    x_2 = siv.fit_transform(x2)
    x_12 = siv.fit_transform(x12)
    x_12_r = siv.fit_transform(x12r)

    """
    # Visualize smoothing
    import matplotlib.pyplot as plt
    plt.plot(x12r[0])
    plt.plot(x_12_r[0])
    plt.savefig(fname=f'SIV_{window_length}.png')
    plt.clf()
    """

    np.testing.assert_almost_equal(x_1[0], x_12[0], decimal=4)
    np.testing.assert_almost_equal(x_2[0], x_12[1], decimal=4)
    assert x_12.shape == x_12_r.shape
