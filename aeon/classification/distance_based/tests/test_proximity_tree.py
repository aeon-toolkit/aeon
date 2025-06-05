"""Tests for ProximityTree."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.classification.distance_based import ProximityTree
from aeon.classification.distance_based._proximity_tree import (
    gini,
    gini_gain,
    msm_params,
    twe_lmbda_params,
    twe_nu_params,
)
from aeon.datasets import load_unit_test


def test_gini():
    """Test the method to calculate gini."""
    # Test case: Pure node (all same class)
    y_pure = np.array([1, 1, 1, 1])
    assert gini(y_pure) == 0.0

    # Test case: Impure node with two classes
    y_impure = np.array([1, 1, 2, 2])
    assert gini(y_impure) == 0.5

    # Test case: More impure node with three classes
    y_more_impure = np.array([1, 1, 2, 3])
    gini_score = 1 - ((2 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2)
    assert gini(y_more_impure) == gini_score

    # Test case: All different classes
    y_all_different = np.array([1, 2, 3, 4])
    gini_score_all_diff = 1 - (
        (1 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2
    )
    assert gini(y_all_different) == gini_score_all_diff

    # Test case: Empty array
    y_empty = np.array([])
    with pytest.raises(ValueError, match="y is empty"):
        gini(y_empty)


def test_gini_gain():
    """Test the method to calculate gini gain of a node."""
    # Split with mixed children
    y = np.array([1, 1, 2, 2, 4, 4, 2, 2])
    y_subs = [np.array([1, 1, 4, 4]), np.array([2, 2, 2, 2])]
    score_y = 1 - ((2 / 8) ** 2 + (4 / 8) ** 2 + (2 / 8) ** 2)
    score = score_y - ((4 / 8) * 0.5 + (4 / 8) * 0)
    assert gini_gain(y, y_subs) == score

    # Split with pure children
    y = np.array([1, 1, 0, 0])
    y_children = [np.array([1, 1]), np.array([0, 0], dtype=y.dtype)]
    assert gini_gain(y, y_children) == gini(y)

    # Test case: Empty array
    y_empty = np.array([])
    with pytest.raises(ValueError, match="y is empty"):
        gini(y_empty)

    # When labels in y_subs do not sum to the same as y
    y_empty = np.array([1, 1, 0, 0])
    y_children = [
        np.array([1, 1]),
        np.array(
            [
                0,
            ],
            dtype=y.dtype,
        ),
    ]
    with pytest.raises(ValueError, match="labels in y_subs must sum"):
        gini_gain(y_empty, y_children)


def test_get_candidate_splitter():
    """Test the method to generate candidate splitters."""
    X, y = load_unit_test()
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
    X, y = load_unit_test()
    clf = ProximityTree(n_splitters=3)
    rng = check_random_state(None)
    unique_classes = np.unique(y)

    splitter, split = clf._get_best_splitter(X, X, y, unique_classes, rng)

    assert isinstance(splitter, tuple)
    assert len(splitter) == 3

    assert isinstance(split, list)
    assert len(split) == len(unique_classes)
    assert sum(len(s) for s in split) == len(y)
