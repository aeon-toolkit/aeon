"""Tests for DFT transformation."""

import numpy as np
import pytest

from aeon.transformations.series.smoothing._dfa import DiscreteFourierApproximation


@pytest.mark.parametrize("r", [0.00, 0.50, 1.00])
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

    dft = DiscreteFourierApproximation(r=r, sort=sort)
    x_1 = dft.fit_transform(x1)
    x_2 = dft.fit_transform(x2)
    x_12 = dft.fit_transform(x12)

    np.testing.assert_almost_equal(x_1[0], x_12[0], decimal=4)
    np.testing.assert_almost_equal(x_2[0], x_12[1], decimal=4)


@pytest.mark.parametrize("r", [0.25, 0.50, 1.00])
@pytest.mark.parametrize("sort", [True, False])
def test_dft_preserves_retained_amplitude(r, sort):
    """A sinusoid inside the retained band must come back with its amplitude."""
    n_timepoints = 64
    k = 3  # frequency bin, well inside the retained band for every r above
    n = np.arange(n_timepoints)
    x = 2.0 + np.cos(2 * np.pi * k * n / n_timepoints)

    dft = DiscreteFourierApproximation(r=r, sort=sort)
    x_ = dft.fit_transform(x)[0]

    np.testing.assert_allclose(x_, x, atol=1e-8)


def test_dft_full_r_is_identity():
    """r=1.0 retains every term, so the series must be reconstructed exactly."""
    rng = np.random.RandomState(0)
    for n_timepoints in (63, 64):
        x = rng.normal(size=n_timepoints)
        x_ = DiscreteFourierApproximation(r=1.0).fit_transform(x)[0]
        np.testing.assert_allclose(x_, x, atol=1e-8)


def test_dft_discards_high_frequencies():
    """A term above the cut-off must be removed, not merely attenuated."""
    n_timepoints = 64
    n = np.arange(n_timepoints)
    low = np.cos(2 * np.pi * 2 * n / n_timepoints)
    high = np.cos(2 * np.pi * 30 * n / n_timepoints)

    x_ = DiscreteFourierApproximation(r=0.25, sort=False).fit_transform(low + high)[0]

    np.testing.assert_allclose(x_, low, atol=1e-8)
