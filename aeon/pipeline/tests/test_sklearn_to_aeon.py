"""Tests for the sklearn_to_aeon function."""

import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from aeon.pipeline import sklearn_to_aeon
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestClassifier(n_estimators=5),
        RandomForestRegressor(n_estimators=5),
        KMeans(n_clusters=2, max_iter=10),
        StandardScaler(),
    ],
)
def test_sklearn_to_aeon(estimator):
    """Test that sklearn_to_aeon works for different types of sklearn estimators."""
    X, y = make_example_3d_numpy()

    est = sklearn_to_aeon(estimator)
    est.fit(X, y)

    if hasattr(est, "predict"):
        est.predict(X)
    else:
        est.transform(X)

    assert est._estimator_type == getattr(estimator, "_estimator_type", "transformer")
