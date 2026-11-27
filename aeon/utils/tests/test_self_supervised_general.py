"""Tests for self supervised utilities."""

import numpy as np
import pytest

from aeon.utils.self_supervised.general import z_normalization


@pytest.mark.parametrize("value", [0.1, 1 / 3, 7.7])
def test_constant_series_normalizes_to_zero(value):
    """A constant series is normalized to zeros whatever its float value."""
    X = np.full((2, 10, 1), value)
    np.testing.assert_allclose(z_normalization(X), 0, atol=1e-12)


def test_z_normalization_zero_mean_unit_std():
    """Each series is normalized to zero mean and unit std along the time axis."""
    X = np.random.default_rng(0).random((3, 20, 2))
    out = z_normalization(X)
    np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=1), 1)
