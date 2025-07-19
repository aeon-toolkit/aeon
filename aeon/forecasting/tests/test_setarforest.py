"""Test the setar-forest forecaster."""

import numpy as np

from aeon.forecasting import SETARForest, SetartreeForecaster


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
    assert all(isinstance(tree, SetartreeForecaster) for tree in forest.estimators_)
    assert all(tree.is_fitted for tree in forest.estimators_)


def test_forest_end_to_end_run():
    """Ensure the forest can fit and predict without errors."""
    forest = SETARForest(
        n_estimators=2,
        lag=3,
        horizon=1,
        bagging_fraction=0.5,
    )

    y_panel = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9, 10, 11],
            [8, 9, 10, 11, 12, 13, 14],
        ]
    )

    y_fit = y_panel[0]
    exog_fit = y_panel[1:]

    # Fit the model
    forest.fit(y_fit, exog=exog_fit)

    # Make a prediction
    history_for_pred = y_panel[0, -5:]  # Use last 5 points of a series
    prediction = forest.predict(y=history_for_pred)

    # Check that the prediction is a valid number
    assert isinstance(prediction, float)
    assert not np.isnan(prediction)
    assert not np.isinf(prediction)
