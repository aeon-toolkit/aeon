"""Tests for BA."""

import re

import numpy as np
import pytest

from aeon.clustering.averaging import (
    elastic_barycenter_average,
    petitjean_barycenter_average,
)
from aeon.clustering.averaging._ba_utils import _get_init_barycenter
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.testing.expected_results.expected_average_results import (
    expected_petitjean_ba_multivariate,
    expected_petitjean_ba_univariate,
)
from aeon.testing.testing_config import MULTITHREAD_TESTING

_average_distances_with_params = [
    ("dtw", {"window": 0.2}),
    ("wdtw", {"g": 2.0}),
    ("edr", {"epsilon": 0.8}),
    ("twe", {"nu": 0.2, "lmbda": 0.1}),
    ("msm", {"c": 2.0}),
    ("shape_dtw", {"reach": 4}),
    ("adtw", {"warp_penalty": 0.2}),
    ("soft_dtw", {"gamma": 0.1}),
]


def test_petitjean_ba_expected():
    """Test petitjean expect result."""
    X_train_uni = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)

    average_ts_uni = elastic_barycenter_average(
        X_train_uni, method="petitjean", random_state=1
    )
    call_directly_average_ts_uni = petitjean_barycenter_average(
        X_train_uni, random_state=1
    )
    assert isinstance(average_ts_uni, np.ndarray)
    assert average_ts_uni.shape == X_train_uni[0].shape
    assert np.allclose(average_ts_uni, expected_petitjean_ba_univariate)
    assert np.allclose(average_ts_uni, call_directly_average_ts_uni)

    X_train_multi = make_example_3d_numpy(10, 3, 10, random_state=1, return_y=False)

    average_ts_multi = elastic_barycenter_average(
        X_train_multi, method="petitjean", random_state=1
    )
    call_directly_average_ts_multi = petitjean_barycenter_average(
        X_train_multi, random_state=1
    )

    assert isinstance(average_ts_multi, np.ndarray)
    assert average_ts_multi.shape == X_train_multi[0].shape
    assert np.allclose(average_ts_multi, expected_petitjean_ba_multivariate)
    assert np.allclose(average_ts_multi, call_directly_average_ts_multi)


@pytest.mark.parametrize("distance", _average_distances_with_params)
@pytest.mark.parametrize(
    "init_barycenter",
    [
        "mean",
        "medoids",
        "random",
        make_example_1d_numpy(10, random_state=5),
        make_example_2d_numpy_series(n_timepoints=10, n_channels=1, random_state=5),
    ],
)
def test_petitjean_ba_uni(distance, init_barycenter):
    """Test petitjean dba functionality."""
    distance = distance[0]
    X_train_uni = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)

    params = {
        "window": 0.2,
        "random_state": 1,
        "init_barycenter": init_barycenter,
        "distance": distance,
    }

    average_ts_uni = elastic_barycenter_average(
        X_train_uni,
        method="petitjean",
        **params,
    )
    call_directly_average_ts_uni = petitjean_barycenter_average(
        X_train_uni,
        **params,
    )
    init_barycenter = _get_init_barycenter(
        X_train_uni,
        **params,
    )

    assert isinstance(average_ts_uni, np.ndarray)
    assert average_ts_uni.shape == X_train_uni[0].shape
    assert np.allclose(average_ts_uni, call_directly_average_ts_uni)
    # EDR and shape_dtw with random values don't update the barycenter so skipping
    if distance not in ["shape_dtw"] and (
        isinstance(init_barycenter, str) and init_barycenter != "mean"
    ):
        # Test not just returning the init barycenter
        assert not np.array_equal(average_ts_uni, init_barycenter)


@pytest.mark.parametrize("distance", _average_distances_with_params)
@pytest.mark.parametrize(
    "init_barycenter",
    [
        "mean",
        "medoids",
        "random",
        make_example_2d_numpy_series(n_timepoints=10, n_channels=3, random_state=5),
    ],
)
def test_petitjean_ba_multi(distance, init_barycenter):
    """Test petitjean multivariate functionality."""
    distance = distance[0]
    X_train_multi = make_example_3d_numpy(10, 3, 10, random_state=1, return_y=False)

    params = {
        "window": 0.2,
        "random_state": 1,
        "init_barycenter": init_barycenter,
        "distance": distance,
    }

    average_ts_multi = elastic_barycenter_average(
        X_train_multi,
        method="petitjean",
        **params,
    )
    call_directly_average_ts_multi = petitjean_barycenter_average(
        X_train_multi,
        **params,
    )
    init_barycenter = _get_init_barycenter(
        X_train_multi,
        **params,
    )

    assert isinstance(average_ts_multi, np.ndarray)
    assert average_ts_multi.shape == X_train_multi[0].shape
    assert np.allclose(average_ts_multi, call_directly_average_ts_multi)
    # EDR and shape_dtw with random values don't update the barycenter so skipping
    if distance not in ["edr", "shape_dtw"] and (
        isinstance(init_barycenter, str) and init_barycenter != "mean"
    ):
        # Test not just returning the init barycenter
        assert not np.array_equal(average_ts_multi, init_barycenter)


@pytest.mark.parametrize("distance", _average_distances_with_params)
def test_petitjean_distance_params(distance):
    """Test petitjean with various distance parameters."""
    distance_params = distance[1]
    distance = distance[0]
    X_train_uni = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)

    for key in distance_params:
        curr_param = {key: distance_params[key]}
        average_ts_uni = elastic_barycenter_average(
            X_train_uni,
            method="petitjean",
            random_state=1,
            distance=distance,
            **curr_param,
        )
        call_directly_average_ts_uni = petitjean_barycenter_average(
            X_train_uni, random_state=1, distance=distance, **curr_param
        )
        assert isinstance(average_ts_uni, np.ndarray)
        assert average_ts_uni.shape == X_train_uni[0].shape
        assert np.allclose(average_ts_uni, call_directly_average_ts_uni)

        no_params_average = petitjean_barycenter_average(
            X_train_uni, random_state=1, distance=distance
        )

        assert not np.array_equal(average_ts_uni, no_params_average)


def test_petitjean_ba_weights():
    """Test weight parameter."""
    num_cases = 4
    X_train_uni = make_example_3d_numpy(
        num_cases, 1, 10, random_state=1, return_y=False
    )
    ones_weights = np.ones(num_cases)
    np.random.seed(1)
    random_weights = np.random.rand(num_cases)

    ba_no_weight_uni = petitjean_barycenter_average(X_train_uni, random_state=1)

    ba_weights_ones_uni = petitjean_barycenter_average(
        X_train_uni, weights=ones_weights, random_state=1
    )

    ba_weights_random_uni = petitjean_barycenter_average(
        X_train_uni, weights=random_weights, random_state=1
    )

    assert np.array_equal(ba_no_weight_uni, ba_weights_ones_uni)
    assert np.array_equal(
        ba_weights_random_uni,
        elastic_barycenter_average(
            X_train_uni, weights=random_weights, random_state=1, method="petitjean"
        ),
    )
    assert not np.array_equal(ba_no_weight_uni, ba_weights_random_uni)

    X_train_multi = make_example_3d_numpy(
        num_cases, 4, 10, random_state=1, return_y=False
    )

    ba_no_wight_multi = petitjean_barycenter_average(X_train_multi, random_state=1)

    ba_weights_ones_multi = petitjean_barycenter_average(
        X_train_multi, weights=ones_weights, random_state=1
    )

    ba_weights_random_multi = petitjean_barycenter_average(
        X_train_multi, weights=random_weights, random_state=1
    )

    assert np.array_equal(ba_no_wight_multi, ba_weights_ones_multi)
    assert np.array_equal(
        ba_weights_random_multi,
        elastic_barycenter_average(
            X_train_multi, weights=random_weights, random_state=1, method="petitjean"
        ),
    )
    assert not np.array_equal(ba_no_wight_multi, ba_weights_random_multi)


def test_petitjean_ba_incorrect_input():
    """Test dba incorrect input."""
    # Test invalid distance
    X = make_example_3d_numpy(10, 1, 10, return_y=False)
    with pytest.raises(ValueError, match="Distance parameter invalid"):
        petitjean_barycenter_average(X, distance="Distance parameter invalid")

    # Test invalid init barycenter string
    with pytest.raises(
        ValueError,
        match="init_barycenter string is invalid. Please use one of the "
        "following: 'mean', 'medoids', 'random'",
    ):
        petitjean_barycenter_average(X, init_barycenter="init parameter invalid")

    # Test invalid init barycenter type
    with pytest.raises(
        ValueError,
        match="init_barycenter parameter is invalid. It must either be a "
        "str or a np.ndarray",
    ):
        petitjean_barycenter_average(X, init_barycenter=[[1, 2, 3]])

    # Test invalid init barycenter with wrong shape
    with pytest.raises(
        ValueError,
        match=re.escape(
            "init_barycenter shape is invalid. Expected (1, 10) but " "got (1, 9)"
        ),
    ):
        petitjean_barycenter_average(X, init_barycenter=make_example_1d_numpy(9))


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("distance", _average_distances_with_params)
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_petitjean_threaded(distance, n_jobs):
    """Test petitjean threaded functionality."""
    distance = distance[0]
    data = make_example_3d_numpy(10, 3, 10, random_state=2, return_y=False)
    serial = petitjean_barycenter_average(
        data, distance=distance, n_jobs=1, random_state=1
    )
    parallel = petitjean_barycenter_average(
        data, distance=distance, n_jobs=n_jobs, random_state=1
    )
    utils_parallel = elastic_barycenter_average(
        data, method="petitjean", distance=distance, n_jobs=n_jobs, random_state=1
    )
    assert serial.shape == parallel.shape
    assert np.allclose(serial, parallel)
    assert np.allclose(serial, utils_parallel)
