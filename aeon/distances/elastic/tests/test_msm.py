"""Tests for MSM distance."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import msm_cost_matrix, msm_distance
from aeon.distances.elastic._msm import _cost_dependent


def _reference_cost_dependent(x, y, z, c):
    """Previous NumPy formulation of the dependent MSM split/merge cost."""
    diameter = np.sum((y - z) ** 2)
    distance_to_mid = np.sum(((y + z) / 2.0 - x) ** 2)
    if distance_to_mid <= diameter / 4.0:
        return c
    dist_to_q_prev = np.sum((y - x) ** 2)
    dist_to_c = np.sum((z - x) ** 2)
    if dist_to_q_prev < dist_to_c:
        return c + dist_to_q_prev
    return c + dist_to_c


@pytest.mark.parametrize("n_channels", [1, 3, 10])
def test_cost_dependent_matches_reference(n_channels):
    """_cost_dependent is numerically equivalent to the previous formulation."""
    rng = np.random.default_rng(n_channels)
    for _ in range(200):
        x, y, z = rng.normal(scale=rng.uniform(0.1, 10.0), size=(3, n_channels))
        assert_allclose(
            _cost_dependent(x, y, z, 1.0),
            _reference_cost_dependent(x, y, z, 1.0),
            rtol=1e-12,
        )


def test_dependent_msm_rejects_different_channel_counts():
    """Dependent MSM raises a ValueError when channel counts differ."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 20))
    y = rng.normal(size=(2, 25))
    match = "same number of channels"

    with pytest.raises(ValueError, match=match):
        msm_distance(x, y, independent=False)
    with pytest.raises(ValueError, match=match):
        msm_cost_matrix(x, y, independent=False)
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match=match):
        msm_distance(x, y, independent=False, window=0.5)
