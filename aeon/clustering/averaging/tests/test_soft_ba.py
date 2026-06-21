"""Tests for gradient-based soft barycentre averaging."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.clustering.averaging import (
    elastic_barycenter_average,
    soft_barycenter_average,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.utils._distance_parameters import TEST_SOFT_DISTANCES_WITH_PARAMS


@pytest.mark.parametrize("distance,params", TEST_SOFT_DISTANCES_WITH_PARAMS)
@pytest.mark.parametrize("n_channels", [1, 3])
def test_soft_ba_shape_and_finite(distance, params, n_channels):
    """Soft barycentre has the series shape and finite values, uni/multivariate."""
    X = make_example_3d_numpy(
        n_cases=6,
        n_channels=n_channels,
        n_timepoints=12,
        return_y=False,
        random_state=1,
    )
    bary = soft_barycenter_average(X, distance=distance, **params)
    assert bary.shape == (n_channels, 12)
    assert np.all(np.isfinite(bary))


@pytest.mark.parametrize("distance,params", TEST_SOFT_DISTANCES_WITH_PARAMS)
def test_soft_ba_return_cost_and_distances(distance, params):
    """``return_cost`` / ``return_distances_to_center`` give finite values."""
    X = make_example_3d_numpy(
        n_cases=5, n_channels=2, n_timepoints=10, return_y=False, random_state=2
    )
    bary, dists, cost = soft_barycenter_average(
        X,
        distance=distance,
        return_distances_to_center=True,
        return_cost=True,
        **params,
    )
    assert dists.shape == (5,)
    # Distances are recomputed at the optimised barycentre (not the inf init).
    assert np.all(np.isfinite(dists))
    assert np.isfinite(cost)


@pytest.mark.parametrize("distance,params", TEST_SOFT_DISTANCES_WITH_PARAMS)
def test_soft_ba_weights_change_result(distance, params):
    """Non-uniform weights should change the barycentre."""
    X = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=10, return_y=False, random_state=3
    )
    unweighted = soft_barycenter_average(X, distance=distance, **params)
    weights = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weighted = soft_barycenter_average(X, distance=distance, weights=weights, **params)
    assert not np.allclose(unweighted, weighted)


def test_soft_ba_single_series_returned_unchanged():
    """A single-series collection returns that series."""
    X = make_example_3d_numpy(
        n_cases=1, n_channels=1, n_timepoints=8, return_y=False, random_state=4
    )
    bary = soft_barycenter_average(X, distance="soft_dtw")
    assert_allclose(bary, X[0])


def test_soft_ba_only_soft_distances_via_method_soft():
    """``method="soft"`` requires a soft distance; discrete methods reject soft."""
    X = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, return_y=False, random_state=5
    )
    with pytest.raises(ValueError, match="requires a soft distance"):
        elastic_barycenter_average(X, method="soft", distance="dtw")
    with pytest.raises(ValueError, match="only be averaged with"):
        elastic_barycenter_average(X, method="petitjean", distance="soft_dtw")


@pytest.mark.parametrize("distance,params", TEST_SOFT_DISTANCES_WITH_PARAMS)
def test_soft_ba_via_elastic_dispatch(distance, params):
    """``elastic_barycenter_average(method='soft')`` matches the direct call."""
    X = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, return_y=False, random_state=6
    )
    direct = soft_barycenter_average(X, distance=distance, **params)
    dispatched = elastic_barycenter_average(
        X, method="soft", distance=distance, **params
    )
    assert_allclose(direct, dispatched)


def test_soft_dtw_barycenter_matches_tslearn():
    """soft-DTW barycentre should match tslearn's ``softdtw_barycenter``.

    Both minimise the same smooth soft-DTW objective with L-BFGS from the mean
    initialisation, so the optima should agree closely.
    """
    ts_bary = pytest.importorskip("tslearn.barycenters")
    X = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=12, return_y=False, random_state=7
    )
    gamma = 1.0
    aeon_bary = soft_barycenter_average(
        X, distance="soft_dtw", gamma=gamma, max_iters=100, tol=1e-6
    )
    # tslearn expects (n_cases, n_timepoints, n_channels); aeon uses
    # (n_cases, n_channels, n_timepoints).
    X_tslearn = np.transpose(X, (0, 2, 1))
    tslearn_bary = ts_bary.softdtw_barycenter(X_tslearn, gamma=gamma, max_iter=100)
    assert_allclose(aeon_bary, tslearn_bary.T, rtol=1e-2, atol=1e-2)
