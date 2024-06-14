"""Tests for ProximityTree."""

import numpy as np
import pytest

from aeon.classification.distance_based import ProximityTree
from aeon.classification.distance_based._proximity_tree import gini, gini_gain


@pytest.fixture
def sample_data():
    """Generate some sample data."""
    X = np.random.rand(10, 50)
    return X


@pytest.fixture
def sample_labels():
    """Generate sample labels for the sample data."""
    y = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0, 1])
    return y


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
    with pytest.raises(ValueError, match="y empty"):
        gini(y_empty)


def test_gini_gain():
    """Test the method to calculate gini gain of a node."""
    # Split with non-empty children
    y = np.array([1, 1, 2, 2, 4, 4, 2, 2])
    y_subs = [np.array([1, 1, 4, 4]), np.array([2, 2, 2, 2])]
    score_y = 1 - ((2 / 8) ** 2 + (4 / 8) ** 2 + (2 / 8) ** 2)
    score = score_y - ((4 / 8) * 0.5 + (4 / 8) * 0)
    assert gini_gain(y, y_subs) == score

    # Split with an empty child
    y = np.array([1, 1, 0, 0])
    y_children = [np.array([1, 1]), np.array([], dtype="int32")]
    score = 0.5 - ((1 / 2) * 0)
    assert gini_gain(y, y_children) == score


def test_get_parameter_value(sample_data):
    """Test the distance parameters generated."""
    X = sample_data
    random_state = 42
    tree = ProximityTree(random_state=random_state)

    params = tree.get_parameter_value(X)

    # Check if the parameters are generated for all distance measures
    expected_measures = [
        "euclidean",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "lcss",
        "twe",
        "msm",
    ]
    assert set(params.keys()) == set(expected_measures)

    # Check specific parameter ranges
    for measure, measure_params in params.items():
        if measure in ["dtw", "ddtw", "lcss"]:
            assert 0 <= measure_params["window"] <= 0.25
        elif measure in ["wdtw", "wddtw"]:
            assert 0 <= measure_params["g"] <= 1
        elif measure == "erp":
            X_std = X.std()
            assert X_std / 5 <= measure_params["g"] <= X_std
        elif measure == "lcss":
            X_std = X.std()
            assert X_std / 5 <= measure_params["epsilon"] <= X_std
        elif measure == "twe":
            assert 0 <= measure_params["lmbda"] < 9
            assert 1e-5 <= measure_params["nu"] <= 1e-1
        elif measure == "msm":
            assert measure_params["c"] in [10**i for i in range(-2, 3)]


def test_get_cadidate_splitter(sample_data, sample_labels):
    """Test the method to generate candidate splitters."""
    X = sample_data
    y = sample_labels
    clf = ProximityTree()
    splitter = clf.get_candidate_splitter(X, y)
    assert len(splitter) == 2

    expected_measures = [
        "euclidean",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "lcss",
        "twe",
        "msm",
    ]
    measure = list(splitter[1].keys())[0]
    assert measure in expected_measures


def test_get_best_splitter(sample_data, sample_labels):
    """Test the method to get optimum splitter of a node."""
    X = sample_data
    y = sample_labels
    clf = ProximityTree(n_splitters=3)

    splitter = clf.get_best_splitter(X, y)

    assert splitter is not None

    assert isinstance(splitter, list)

    assert len(splitter) == 2
