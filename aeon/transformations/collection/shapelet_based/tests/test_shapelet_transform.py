"""Tests for RandomShapeletTransform helpers."""

import numpy as np

from aeon.transformations.collection.shapelet_based._shapelet_transform import (
    _online_shapelet_distance,
)
from aeon.utils.numba.general import z_normalise_series


def test_online_shapelet_distance_offset_invariance():
    """z-normalised shapelet distance should be invariant to constant offset.

    Regression test for GH #3643: rolling variance used an unstable formula
    that cancelled for high-offset, low-variance series, so adding a constant
    could change the distance from ~0 to ~1.
    """
    rng = np.random.RandomState(0)
    position = 30
    length = 20

    # Unit-scale series: float64 preserves the signal after adding a large offset.
    series = rng.normal(size=100)
    shapelet = z_normalise_series(series[position : position + length])
    sorted_indices = np.argsort(np.abs(shapelet))[::-1].astype(np.int32)

    base = _online_shapelet_distance(series, shapelet, sorted_indices, position, length)
    shifted = _online_shapelet_distance(
        series + 1e8, shapelet, sorted_indices, position, length
    )
    assert np.allclose(shifted, base, rtol=1e-8, atol=1e-8)

    # High-offset, low-variance series from the issue report. Exact equality
    # is limited by float64 quantisation of series+1e8, but the distance must
    # stay near zero rather than collapsing to ~1.
    series = rng.normal(scale=1e-5, size=100)
    shapelet = z_normalise_series(series[position : position + length])
    sorted_indices = np.argsort(np.abs(shapelet))[::-1].astype(np.int32)

    base = _online_shapelet_distance(series, shapelet, sorted_indices, position, length)
    shifted = _online_shapelet_distance(
        series + 1e8, shapelet, sorted_indices, position, length
    )
    assert base < 1e-6
    assert shifted < 1e-3
