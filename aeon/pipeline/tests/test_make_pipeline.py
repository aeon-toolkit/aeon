"""Tests for the make_pipeline function."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from aeon.base import BaseEstimator
from aeon.classification import DummyClassifier
from aeon.clustering import TimeSeriesKMeans
from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.naive import NaiveForecaster
from aeon.pipeline import make_pipeline
from aeon.regression import DummyRegressor
from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
)
from aeon.testing.mock_estimators import MockTransformer
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import PaddingTransformer, Tabularizer
from aeon.transformations.collection.feature_based import SevenNumberSummaryTransformer


@pytest.mark.parametrize(
    "pipeline",
    [
        [PaddingTransformer(pad_length=15), DummyClassifier()],
        [SevenNumberSummaryTransformer(), RandomForestClassifier(n_estimators=2)],
        [PaddingTransformer(pad_length=15), DummyRegressor()],
        [SevenNumberSummaryTransformer(), RandomForestRegressor(n_estimators=2)],
        [PaddingTransformer(pad_length=15), TimeSeriesKMeans.create_test_instance()],
        [SevenNumberSummaryTransformer(), KMeans(n_clusters=2, max_iter=3)],
        [PaddingTransformer(pad_length=15), SevenNumberSummaryTransformer()],
        [PaddingTransformer(pad_length=15), Tabularizer(), StandardScaler()],
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

    assert isinstance(est, BaseEstimator)
    assert isinstance(o, np.ndarray)


@pytest.mark.parametrize(
    "pipeline",
    [
        [MockTransformer(), NaiveForecaster()],
        [MockTransformer(), MockTransformer(power=3)],
    ],
)
def test_make_pipeline_legacy(pipeline):
    """Test that make_pipeline works for some legacy interfaces."""
    assert isinstance(pipeline[-1], BaseTransformer) or isinstance(
        pipeline[-1], BaseForecaster
    )

    X, y = make_example_2d_numpy_collection()

    est = make_pipeline(pipeline)
    est.fit(X, y)

    if hasattr(est, "predict"):
        o = est.predict([0, 1, 2, 3])
    else:
        o = est.transform(X)

    assert isinstance(est, BaseEstimator)
    assert isinstance(o, np.ndarray)
