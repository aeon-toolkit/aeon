"""Tests for shared options of the barycenter averaging functions.

These cover the pure-Python control flow shared by the Petitjean, subgradient and
KASBA barycenter averagers: the single-case early return, the ``return_cost`` /
``return_distances_to_center`` output options and the ``verbose`` prints. The inner
per-iteration update loops are numba-compiled and therefore exercised through the
existing per-method functional tests.
"""

import numpy as np
import pytest

from aeon.clustering.averaging import (
    kasba_average,
    petitjean_barycenter_average,
    subgradient_barycenter_average,
)
from aeon.testing.data_generation import make_example_3d_numpy

BA_FUNCTIONS = [
    petitjean_barycenter_average,
    subgradient_barycenter_average,
    kasba_average,
]


@pytest.mark.parametrize("ba_func", BA_FUNCTIONS)
def test_ba_single_case_early_return(ba_func):
    """Test the early return for a single time series and all output options."""
    X = make_example_3d_numpy(1, 1, 8, return_y=False, random_state=1)

    center = ba_func(X)
    assert center.shape == (1, 8)
    assert np.array_equal(center, X[0])

    center, dists = ba_func(X, return_distances_to_center=True)
    assert np.array_equal(center, X[0])
    assert np.array_equal(dists, np.zeros(1))

    center, cost = ba_func(X, return_cost=True)
    assert np.array_equal(center, X[0])
    assert cost == 0.0

    center, dists, cost = ba_func(X, return_distances_to_center=True, return_cost=True)
    assert np.array_equal(center, X[0])
    assert np.array_equal(dists, np.zeros(1))
    assert cost == 0.0


@pytest.mark.parametrize("ba_func", BA_FUNCTIONS)
def test_ba_return_options(ba_func):
    """Test the return_cost and return_distances_to_center options on real data."""
    X = make_example_3d_numpy(8, 1, 8, return_y=False, random_state=1)
    kwargs = {"distance": "dtw", "random_state": 1, "max_iters": 5}

    center = ba_func(X, **kwargs)
    assert center.shape == (1, 8)

    center, dists = ba_func(X, return_distances_to_center=True, **kwargs)
    assert center.shape == (1, 8)
    assert dists.shape == (8,)

    center, cost = ba_func(X, return_cost=True, **kwargs)
    assert center.shape == (1, 8)
    assert isinstance(float(cost), float)

    center, dists, cost = ba_func(
        X, return_distances_to_center=True, return_cost=True, **kwargs
    )
    assert center.shape == (1, 8)
    assert dists.shape == (8,)
    assert isinstance(float(cost), float)


@pytest.mark.parametrize("ba_func", BA_FUNCTIONS)
def test_ba_verbose(ba_func, capsys):
    """Test the barycenter averagers print progress when verbose is True."""
    X = make_example_3d_numpy(8, 1, 8, return_y=False, random_state=1)
    ba_func(X, distance="dtw", random_state=1, max_iters=5, verbose=True)
    out = capsys.readouterr().out
    assert "epoch" in out
