# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the BaseIntervalForest class."""

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.interval_based._interval_forest import IntervalForestClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.transformations.collection import (
    AutocorrelationFunctionTransformer,
    Catch22,
    SevenNumberSummaryTransformer,
)
from aeon.transformations.series.func_transform import FunctionTransformer
from aeon.utils._testing.collection import make_3d_test_data
from aeon.utils.numba.stats import row_mean, row_numba_min


@pytest.mark.parametrize(
    "base_estimator",
    [DecisionTreeClassifier(), ContinuousIntervalTree()],
)
def test_interval_forest_feature_skipping(base_estimator):
    """Test BaseIntervalForest feature skipping with different base estimators."""
    X, y = make_3d_test_data(random_state=0)

    est = IntervalForestClassifier(
        base_estimator=base_estimator,
        n_estimators=2,
        n_intervals=2,
        random_state=0,
    )
    est.fit(X, y)
    preds = est.predict(X)

    assert est._efficient_predictions is True

    est = IntervalForestClassifier(
        base_estimator=make_pipeline(base_estimator),
        n_estimators=2,
        n_intervals=2,
        random_state=0,
    )
    est.fit(X, y)

    assert est._efficient_predictions is False
    assert (preds == est.predict(X)).all()


def test_interval_forest_invalid_feature_skipping():
    """Test BaseIntervalForest with an invalid transformer for feature skipping."""
    X, y = make_3d_test_data()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        interval_features=SevenNumberSummaryTransformer(),
    )
    est.fit(X, y)

    assert est._efficient_predictions is False


@pytest.mark.parametrize(
    "interval_selection_method",
    ["random", "supervised", "random-supervised"],
)
def test_interval_forest_selection_methods(interval_selection_method):
    """Test BaseIntervalForest with different interval selection methods."""
    X, y = make_3d_test_data()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        interval_selection_method=interval_selection_method,
    )
    est.fit(X, y)

    assert est.predict_proba(X).shape == (10, 2)


@pytest.mark.parametrize(
    "n_intervals,n_intervals_len",
    [
        ("sqrt", 24),
        ("sqrt-div", 12),
        (["sqrt-div", 2], 24),
        ([[1, 2], "sqrt-div"], 15),
    ],
)
def test_interval_forest_n_intervals(n_intervals, n_intervals_len):
    """Test BaseIntervalForest n_interval options."""
    X, y = make_3d_test_data(n_timepoints=20)

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=n_intervals,
        series_transformers=[None, FunctionTransformer(np.log1p)],
        save_transformed_data=True,
        random_state=0,
    )
    est.fit(X, y)
    est.predict_proba(X)

    data = est.transformed_data_
    assert data[0].shape[1] == n_intervals_len


att_subsample_c22 = Catch22(
    features=[
        "DN_HistogramMode_5",
        "DN_HistogramMode_10",
        "SB_BinaryStats_diff_longstretch0",
    ]
)


@pytest.mark.parametrize(
    "features,output_len",
    [
        (None, 3),
        (_clone_estimator(att_subsample_c22), 3),
        ([_clone_estimator(att_subsample_c22), _clone_estimator(att_subsample_c22)], 6),
        (
            [
                row_mean,
                Catch22(features=["DN_HistogramMode_5", "DN_HistogramMode_10"]),
                row_numba_min,
            ],
            4,
        ),
    ],
)
def test_interval_forest_attribute_subsample(features, output_len):
    """Test BaseIntervalForest subsampling with different interval features."""
    X, y = make_3d_test_data()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=0.5,
        interval_features=features,
        replace_nan=0,
        save_transformed_data=True,
        random_state=0,
    )
    est.fit(X, y)
    est.predict_proba(X)

    data = est.transformed_data_
    assert data[0].shape[1] == int(output_len * 0.5) * 2


def test_interval_forest_invalid_attribute_subsample():
    """Test BaseIntervalForest with an invalid transformer for subsampling."""
    X, y = make_3d_test_data()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=2,
        interval_features=SevenNumberSummaryTransformer(),
    )

    with pytest.raises(ValueError):
        est.fit(X, y)


@pytest.mark.parametrize(
    "series_transformer",
    [
        FunctionTransformer(np.log1p),
        [None, FunctionTransformer(np.log1p)],
        [FunctionTransformer(np.log1p), AutocorrelationFunctionTransformer(n_lags=6)],
    ],
)
def test_interval_forest_series_transformer(series_transformer):
    """Test BaseIntervalForest with different series transformers."""
    X, y = make_3d_test_data()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        series_transformers=series_transformer,
        save_transformed_data=True,
        random_state=0,
    )
    est.fit(X, y)
    est.predict_proba(X)

    data = est.transformed_data_
    expected = (
        len(series_transformer) * 6 if isinstance(series_transformer, list) else 6
    )
    assert data[0].shape[1] == expected
