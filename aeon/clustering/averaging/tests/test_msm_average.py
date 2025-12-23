"""Tests for MSM barycenter averaging."""

import numpy as np
import pytest

from aeon.clustering.averaging import (
    elastic_barycenter_average,
    msm_barycenter_average,
)
from aeon.testing.data_generation import make_example_3d_numpy


def test_msm_average_shapes():
    """Test that MSM average returns expected shapes."""
    X_uni = make_example_3d_numpy(3, 1, 5, random_state=1, return_y=False)
    avg_uni = msm_barycenter_average(X_uni, c=1.0)
    assert isinstance(avg_uni, np.ndarray)
    assert avg_uni.shape[0] == 1
    assert avg_uni.ndim == 2
    X_multi = make_example_3d_numpy(3, 3, 5, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="only implemented for univariate"):
        msm_barycenter_average(X_multi, c=1.0)


def test_msm_return_distances():
    """Test that MSM returns distances when requested."""
    X = make_example_3d_numpy(3, 1, 5, random_state=1, return_y=False)
    center = msm_barycenter_average(X, c=1.0)
    assert isinstance(center, np.ndarray)
    center, dists = msm_barycenter_average(X, c=1.0, return_distances_to_center=True)
    assert isinstance(center, np.ndarray)
    assert isinstance(dists, np.ndarray)
    assert len(dists) == 3
    assert np.all(dists >= 0)
    center, dists, cost = msm_barycenter_average(
        X, c=1.0, return_distances_to_center=True, return_cost=True
    )
    assert isinstance(cost, float)
    assert cost > 0


def test_msm_average_integration():
    """Test integration with the generic elastic_barycenter_average dispatcher."""
    X = make_example_3d_numpy(3, 1, 5, random_state=1, return_y=False)
    res_direct = msm_barycenter_average(X, c=0.5)
    res_dispatch = elastic_barycenter_average(X, method="msm", c=0.5)
    assert np.allclose(res_direct, res_dispatch)


def test_msm_average_determinism():
    """Test that the Exact MSM solver is deterministic."""
    X = make_example_3d_numpy(3, 1, 5, random_state=42, return_y=False)
    run1 = msm_barycenter_average(X, c=1.0)
    run2 = msm_barycenter_average(X, c=1.0)
    assert np.array_equal(run1, run2)


def test_msm_parameter_c():
    """Test that changing the cost parameter 'c' alters the result."""
    X = np.array([[[1.0, 5.0, 1.0]], [[1.0, 1.0, 5.0]]])
    avg_low = msm_barycenter_average(X, c=0.1)
    avg_high = msm_barycenter_average(X, c=100.0)
    assert not np.array_equal(avg_low, avg_high)


def test_msm_window_constraint():
    """Test that the window parameter constrains the path."""
    X = np.array([[[0.0, 10.0, 0.0]], [[0.0, 0.0, 10.0]]])
    c_val = 0.1
    avg_no_win = msm_barycenter_average(X, c=c_val, window=None)
    avg_win = msm_barycenter_average(X, c=c_val, window=0)
    assert not np.array_equal(avg_no_win, avg_win)
    assert avg_win.size > 0


def test_msm_identity():
    """Test that the average of identical series is the series itself."""
    X = np.array([[[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]])
    avg = msm_barycenter_average(X)
    assert np.array_equal(avg, X[0])


def test_msm_exact_property_subset_values():
    """Test MSM property: Output values are a subset of input values."""
    X = np.array([[[10, 20, 30]], [[20, 30, 40]]])
    avg = msm_barycenter_average(X, c=1.0)
    unique_in = np.unique(X)
    unique_out = np.unique(avg)
    assert np.all(np.isin(unique_out, unique_in))


def test_msm_api_consistency():
    """Test that unused API parameters do not cause crashes."""
    X = make_example_3d_numpy(3, 1, 5, random_state=1, return_y=False)
    try:
        msm_barycenter_average(
            X, weights=np.ones(3), n_jobs=2, init_barycenter="mean", verbose=True
        )
    except Exception as e:
        pytest.fail(f"API parameter caused crash: {e}")
