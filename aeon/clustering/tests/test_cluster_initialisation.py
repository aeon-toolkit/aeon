"""Tests for cluster initialisation functions."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.random import RandomState

from aeon.clustering._cluster_initialisation import (
    _CENTRE_INITIALISER_INDEXES,
    _CENTRE_INITIALISERS,
    _kmeans_plus_plus_center_initialiser_indexes,
    _kmedoids_plus_plus_center_initialiser_indexes,
    resolve_center_initialiser,
)
from aeon.distances._distance import ELASTIC_DISTANCES, POINTWISE_DISTANCES
from aeon.testing.data_generation import make_example_3d_numpy

NON_RANDOM_INIT = ["first"]


def _run_initialisation_test(
    key: str,
    init_func: Callable,
    init_func_indexes: Callable = None,
    init_func_params=None,
):
    if init_func_params is None:
        init_func_params = {}

    X = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    n_clusters = 3
    init_func_params = {
        "X": X,
        "n_clusters": n_clusters,
        **init_func_params,
    }

    values = init_func(**init_func_params, random_state=RandomState(1))

    assert len(values) == n_clusters
    assert values.shape[1:] == X.shape[1:]

    assert np.allclose(
        values, init_func(**init_func_params, random_state=RandomState(1))
    )

    if key not in NON_RANDOM_INIT:
        diff_random_state_values = init_func(
            **init_func_params, random_state=RandomState(2)
        )
        assert not np.allclose(values, diff_random_state_values)

    if init_func_indexes:
        indexes = init_func_indexes(**init_func_params, random_state=RandomState(1))
        value_from_indexes = X[indexes]
        assert np.allclose(values, value_from_indexes)


@pytest.mark.parametrize("init_key", _CENTRE_INITIALISERS.keys())
def test_center_initialisers(init_key):
    """Test all center initialisers."""
    params = {}
    if init_key == "kmeans++" or init_key == "kmedoids++":
        params["distance"] = "euclidean"
        params["distance_params"] = {}

    _run_initialisation_test(
        key=init_key,
        init_func=_CENTRE_INITIALISERS[init_key],
        init_func_indexes=_CENTRE_INITIALISER_INDEXES.get(init_key, None),
        init_func_params=params,
    )


@pytest.mark.parametrize("init_key", ["kmeans++", "kmedoids++"])
@pytest.mark.parametrize("dist", POINTWISE_DISTANCES + ELASTIC_DISTANCES)
def test_distance_center_initialisers(init_key, dist):
    """Test all center initialisers with distance."""
    params = {
        "distance": dist,
        "distance_params": {},
    }
    _run_initialisation_test(
        key=init_key,
        init_func=_CENTRE_INITIALISERS[init_key],
        init_func_indexes=_CENTRE_INITIALISER_INDEXES.get(init_key, None),
        init_func_params=params,
    )


@pytest.mark.parametrize("init_key", ["kmeans++", "kmedoids++"])
def test_distance_center_initialisers_params(init_key):
    """Test all center initialisers with distance."""
    n_clusters = 3
    X = make_example_3d_numpy(50, 1, 10, random_state=1, return_y=False)

    init_func_no_window = _CENTRE_INITIALISERS[init_key](
        X=X,
        n_clusters=n_clusters,
        distance_params={},
        distance="soft_dtw",
        random_state=RandomState(1),
    )

    init_func_window = _CENTRE_INITIALISERS[init_key](
        X=X,
        n_clusters=n_clusters,
        distance_params={"gamma": 0.00001},
        distance="soft_dtw",
        random_state=RandomState(1),
    )

    assert not np.array_equal(init_func_no_window, init_func_window)


@pytest.mark.parametrize(
    "init_func",
    [
        _kmeans_plus_plus_center_initialiser_indexes,
        _kmedoids_plus_plus_center_initialiser_indexes,
    ],
)
def test_plus_plus_identical_series(init_func):
    """Test ++ initialisers fall back to random choice when all distances are equal.

    With identical series every candidate distance is zero, so the
    distance-weighted sampling has zero total weight and the initialiser must
    still return ``n_clusters`` distinct indices.
    """
    X = np.ones((6, 1, 5))
    indexes = init_func(
        X=X,
        n_clusters=3,
        random_state=RandomState(0),
        distance="euclidean",
        distance_params={},
    )
    assert len(indexes) == 3
    assert len(set(indexes.tolist())) == 3


def test_resolve_requires_distance_for_plus_plus():
    """Test resolve raises if distance is missing for ++ initialisers."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="distance and distance_params are required"):
        resolve_center_initialiser(
            init="kmeans++",
            X=X,
            n_clusters=2,
            random_state=RandomState(0),
            distance=None,
            distance_params=None,
        )


def test_resolve_custom_init_handler():
    """Test resolve returns a registered custom init handler."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)

    def my_handler(X):
        return np.arange(2)

    resolved = resolve_center_initialiser(
        init="build",
        X=X,
        n_clusters=2,
        random_state=RandomState(0),
        custom_init_handlers={"build": my_handler},
    )
    assert resolved is my_handler


def test_resolve_invalid_init_string():
    """Test resolve raises for an unknown init string."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="is invalid"):
        resolve_center_initialiser(
            init="not_a_real_init",
            X=X,
            n_clusters=2,
            random_state=RandomState(0),
        )


def test_resolve_array_wrong_length():
    """Test resolve raises when the init array length does not match n_clusters."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="Expected length"):
        resolve_center_initialiser(
            init=np.array([0, 1, 2]),
            X=X,
            n_clusters=2,
            random_state=RandomState(0),
            use_indexes=True,
        )


def test_resolve_array_indexes_must_be_1d():
    """Test resolve rejects a multi-dimensional index array when use_indexes."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="Expected 1D array"):
        resolve_center_initialiser(
            init=np.zeros((2, 1, 8)),
            X=X,
            n_clusters=2,
            random_state=RandomState(0),
            use_indexes=True,
        )


def test_resolve_array_centres_must_be_multidimensional():
    """Test resolve rejects a 1D array when centres (not indexes) are expected."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="multi-dimensional"):
        resolve_center_initialiser(
            init=np.array([0, 1]),
            X=X,
            n_clusters=2,
            random_state=RandomState(0),
            use_indexes=False,
        )


def test_resolve_array_centres_wrong_shape():
    """Test resolve rejects a centre array whose shape mismatches X."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="Expected shape"):
        resolve_center_initialiser(
            init=np.zeros((2, 1, 3)),
            X=X,
            n_clusters=2,
            random_state=RandomState(0),
            use_indexes=False,
        )


def test_resolve_invalid_init_type():
    """Test resolve rejects an init that is neither a string nor an array."""
    X = make_example_3d_numpy(10, 1, 8, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="string or np.ndarray"):
        resolve_center_initialiser(
            init=5,
            X=X,
            n_clusters=2,
            random_state=RandomState(0),
        )
