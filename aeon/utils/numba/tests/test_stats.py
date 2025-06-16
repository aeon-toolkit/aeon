"""Tests for numba utils functions related to statistical operations."""

import numpy as np
import pytest

from aeon.utils.numba.stats import gini, gini_gain


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
