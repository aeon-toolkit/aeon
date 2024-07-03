"""Test for Proximity Forest."""

import pytest
from sklearn.metrics import accuracy_score

from aeon.classification.distance_based import ProximityForest
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.fixture
def time_series_dataset():
    """Generate time series dataset for testing."""
    n_samples = 100  # Total number of samples (should be even)
    n_timepoints = 24  # Length of each time series
    n_channels = 1
    data, labels = make_example_3d_numpy(n_samples, n_channels, n_timepoints)
    return data, labels


def test_univariate(time_series_dataset):
    """Test that the function gives appropriate error message."""
    X, y = time_series_dataset
    X_multivariate = X.reshape((100, 2, 12))
    clf = ProximityForest(n_trees=5, random_state=42, n_jobs=-1)
    with pytest.raises(ValueError):
        clf.fit(X_multivariate, y)


def test_proximity_forest(time_series_dataset):
    """Test the fit method of ProximityTree."""
    X, y = time_series_dataset
    clf = ProximityForest(n_trees=5, n_splitters=3, max_depth=4)
    clf.fit(X, y)
    X_test, y_test = time_series_dataset
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    assert score >= 0.9
