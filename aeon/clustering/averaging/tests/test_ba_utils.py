"""Tests for the barycenter averaging utilities in ``_ba_utils``."""

import numpy as np
import pytest

from aeon.clustering.averaging import petitjean_barycenter_average
from aeon.clustering.averaging._ba_utils import _medoids
from aeon.testing.data_generation import make_example_3d_numpy


def test_ba_setup_accepts_2d_input():
    """Test a 2D collection is reshaped and averaged without error."""
    X = np.random.RandomState(0).random((5, 8))
    avg = petitjean_barycenter_average(X, distance="dtw", random_state=1, max_iters=3)
    assert avg.shape == (1, 8)


def test_ba_setup_rejects_invalid_ndim():
    """Test a 1D collection raises an informative error."""
    X = np.arange(8.0)
    with pytest.raises(ValueError, match="X must be a 2D or 3D array"):
        petitjean_barycenter_average(X, distance="dtw")


def test_ba_setup_rejects_mismatched_weights():
    """Test weights that do not match the number of cases raise an error."""
    X = make_example_3d_numpy(5, 1, 8, return_y=False, random_state=1)
    with pytest.raises(ValueError, match="Weights must be the same length as X"):
        petitjean_barycenter_average(
            X, distance="dtw", weights=np.ones(3), random_state=1
        )


def test_medoids_empty_input():
    """Test _medoids returns the input unchanged for an empty collection."""
    empty = np.zeros((0, 1, 8))
    assert _medoids(empty).shape == (0, 1, 8)
