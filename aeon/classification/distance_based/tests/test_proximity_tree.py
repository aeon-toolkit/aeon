"""Tests for ProximityTree."""

import numpy as np
import pytest

from aeon.classification.distance_based import ProximityTree


def test_gini():
    """Test the method to calculate gini."""
    clf = ProximityTree()

    # Test case: Pure node (all same class)
    y_pure = np.array([1, 1, 1, 1])
    assert clf.gini(y_pure) == 0.0

    # Test case: Impure node with two classes
    y_impure = np.array([1, 1, 2, 2])
    assert clf.gini(y_impure) == 0.5

    # Test case: More impure node with three classes
    y_more_impure = np.array([1, 1, 2, 3])
    gini_score = 1 - ((2 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2)
    assert clf.gini(y_more_impure) == gini_score

    # Test case: All different classes
    y_all_different = np.array([1, 2, 3, 4])
    gini_score_all_diff = 1 - (
        (1 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2
    )
    assert clf.gini(y_all_different) == gini_score_all_diff

    # Test case: Empty array
    y_empty = np.array([])
    with pytest.raises(ValueError, match="y empty"):
        clf.gini(y_empty)


def test_gini_gain():
    """Test the method to calculate gini gain of a node."""
    clf = ProximityTree()

    # Split with non-empty children
    y = np.array([1, 1, 2, 2, 4, 4, 2, 2])
    y_subs = [np.array([1, 1, 4, 4]), np.array([2, 2, 2, 2])]
    score_y = 1 - ((2 / 8) ** 2 + (4 / 8) ** 2 + (2 / 8) ** 2)
    score = score_y - ((4 / 8) * 0.5 + (4 / 8) * 0)
    assert clf.gini_gain(y, y_subs) == score

    # Split with an empty child
    y = np.array([1, 1, 0, 0])
    y_children = [np.array([1, 1]), np.array([])]
    score = 0.5 - ((1 / 2) * 0)
    assert clf.gini_gain(y, y_children) == score
