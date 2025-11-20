"""Tests for cluster initialisation functions."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.random import RandomState

from aeon.clustering._cluster_initialisation import (
    CENTER_INITIALISER_INDEXES,
    CENTER_INITIALISERS,
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


@pytest.mark.parametrize("init_key", CENTER_INITIALISERS.keys())
def test_center_initialisers(init_key):
    """Test all center initialisers."""
    params = {}
    if init_key == "kmeans++" or init_key == "kmedoids++":
        params["distance"] = "euclidean"
        params["distance_params"] = {}

    _run_initialisation_test(
        key=init_key,
        init_func=CENTER_INITIALISERS[init_key],
        init_func_indexes=CENTER_INITIALISER_INDEXES.get(init_key, None),
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
        init_func=CENTER_INITIALISERS[init_key],
        init_func_indexes=CENTER_INITIALISER_INDEXES.get(init_key, None),
        init_func_params=params,
    )


@pytest.mark.parametrize("init_key", ["kmeans++", "kmedoids++"])
def test_distance_center_initialisers_params(init_key):
    """Test all center initialisers with distance."""
    n_clusters = 3
    X = make_example_3d_numpy(50, 1, 10, random_state=1, return_y=False)

    init_func_no_window = CENTER_INITIALISERS[init_key](
        X=X,
        n_clusters=n_clusters,
        distance_params={},
        distance="soft_dtw",
        random_state=RandomState(1),
    )

    init_func_window = CENTER_INITIALISERS[init_key](
        X=X,
        n_clusters=n_clusters,
        distance_params={"gamma": 0.00001},
        distance="soft_dtw",
        random_state=RandomState(1),
    )

    assert not np.array_equal(init_func_no_window, init_func_window)
