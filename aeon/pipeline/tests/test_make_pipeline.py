"""Tests for the make_pipeline function."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from aeon.base import BaseAeonEstimator
from aeon.classification import DummyClassifier
from aeon.clustering import TimeSeriesKMeans
from aeon.pipeline import make_pipeline
from aeon.regression import DummyRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import Padder, Tabularizer
from aeon.transformations.collection.feature_based import SevenNumberSummary


@pytest.mark.parametrize(
    "pipeline",
    [
        [Padder(pad_length=15), DummyClassifier()],
        [SevenNumberSummary(), RandomForestClassifier(n_estimators=2)],
        [Padder(pad_length=15), DummyRegressor()],
        [SevenNumberSummary(), RandomForestRegressor(n_estimators=2)],
        [Padder(pad_length=15), TimeSeriesKMeans._create_test_instance()],
        [SevenNumberSummary(), KMeans(n_clusters=2, max_iter=3)],
        [Padder(pad_length=15), SevenNumberSummary()],
        [Padder(pad_length=15), Tabularizer(), StandardScaler()],
    ],
)
def test_make_pipeline(pipeline):
    """Test that make_pipeline works for different types of estimator."""
    X, y = make_example_3d_numpy()

    est = make_pipeline(pipeline)
    est.fit(X, y)

    if hasattr(est, "predict"):
        o = est.predict(X)
    else:
        o = est.transform(X)

    assert isinstance(est, BaseAeonEstimator)
    assert isinstance(o, np.ndarray)
