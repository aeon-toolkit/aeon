"""Test the setar-forest forecaster."""

import numpy as np

from aeon.forecasting import SETARForest, SETARTree


def test_forest_initialization():
    """Test that the forest initializes with correct parameters."""
    forest = SETARForest(n_estimators=5, bagging_fraction=0.7)
    assert forest.n_estimators == 5
    assert forest.bagging_fraction == 0.7
    assert forest.estimators_ == []  # Estimators should be empty before fitting


def test_fit_creates_correct_number_of_trees():
    """Verify that fitting the forest creates the specified number of trees."""
    forest = SETARForest(n_estimators=3)
    # Create a dummy panel with enough series for bagging
    y_panel = np.random.rand(10, 20)
    y_fit = y_panel[0]
    exog_fit = y_panel[1:]

    forest.fit(y_fit, exog=exog_fit)

    assert len(forest.estimators_) == forest.n_estimators
    assert all(isinstance(tree, SETARTree) for tree in forest.estimators_)
    assert all(tree.is_fitted for tree in forest.estimators_)
