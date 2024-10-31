"""Test for Proximity Forest."""

import pytest
from sklearn.metrics import accuracy_score

from aeon.classification.distance_based import ProximityForest
from aeon.datasets import load_unit_test


def test_univariate():
    """Test that the function gives appropriate error message."""
    X, y = load_unit_test()
    X_multivariate = X.reshape((-1, 2, 12))
    clf = ProximityForest(n_trees=5)
    with pytest.raises(ValueError):
        clf.fit(X_multivariate, y)


def test_proximity_forest():
    """Test the fit method of ProximityTree."""
    X_train, y_train = load_unit_test()
    X_test, y_test = load_unit_test(split="test")
    clf = ProximityForest(n_trees=5, n_splitters=3, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    assert score >= 0.9
