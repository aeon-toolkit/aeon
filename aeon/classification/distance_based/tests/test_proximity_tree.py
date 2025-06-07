"""Tests for ProximityTree."""

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.utils import check_random_state

from aeon.classification.distance_based import ProximityTree
from aeon.classification.distance_based._proximity_tree import (
    msm_params,
    twe_lmbda_params,
    twe_nu_params,
)
from aeon.testing.data_generation import make_example_3d_numpy


def test_get_candidate_splitter():
    """Test the method to generate candidate splitters."""
    X, y = make_example_3d_numpy()
    cls_idx = {}
    for label in np.unique(y):
        cls_idx[label] = np.where(y == label)[0]
    clf = ProximityTree()
    rng = check_random_state(0)
    exemplars, distance, distance_params, X_std = clf._get_candidate_splitter(
        X, X, rng, cls_idx, None
    )

    assert isinstance(exemplars, dict)
    assert len(exemplars) == 2
    assert all([isinstance(v, np.ndarray) for v in exemplars.values()])

    expected_distances = [
        "euclidean",
        "dtw-full",
        "dtw",
        "ddtw",
        "ddtw-full",
        "wdtw",
        "wddtw",
        "erp",
        "lcss",
        "twe",
        "msm",
    ]
    assert distance in expected_distances

    if distance in ["dtw", "ddtw", "lcss"]:
        assert 0 <= distance_params["window"] <= 0.25
    elif distance in ["wdtw", "wddtw"]:
        assert 0 <= distance_params["g"] <= 1
    elif distance == "erp":
        assert X_std is not None
        assert X_std / 5 <= distance_params["g"] <= X_std
    elif distance == "lcss":
        assert X_std is not None
        assert X_std / 5 <= distance_params["epsilon"] <= X_std
    elif distance == "twe":
        assert distance_params["lmbda"] in twe_lmbda_params
        assert distance_params["nu"] in twe_nu_params
    elif distance == "msm":
        assert distance_params["c"] in msm_params
    elif distance == "euclidean" or distance == "dtw-full" or distance == "ddtw-full":
        assert distance_params == {}
    else:
        raise ValueError(f"Unexpected distance: {distance}")


def test_get_best_splitter():
    """Test the method to get optimum splitter of a node."""
    X, y = make_example_3d_numpy()
    clf = ProximityTree(n_splitters=3)
    rng = check_random_state(None)
    unique_classes = np.unique(y)

    splitter, split = clf._get_best_splitter(X, X, y, unique_classes, rng)

    assert isinstance(splitter, tuple)
    assert len(splitter) == 3

    assert isinstance(split, list)
    assert len(split) == len(unique_classes)
    assert sum(len(s) for s in split) == len(y)


def test_get_derivatives_and_std():
    """Test the methods to get derivatives and standard deviation."""
    X, y = make_example_3d_numpy()
    X_list = [x for x in X]

    clf = ProximityTree()
    assert_array_equal(clf._get_derivatives(X_list), clf._get_derivatives(X_list))
    assert clf._get_std(X_list) == clf._get_std(X_list)
