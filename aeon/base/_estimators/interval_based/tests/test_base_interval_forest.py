"""Tests for the BaseIntervalForest class."""

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.interval_based._interval_forest import IntervalForestClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import AutocorrelationFunctionTransformer
from aeon.transformations.collection.feature_based import Catch22, SevenNumberSummary
from aeon.utils.numba.stats import row_mean, row_numba_min


@pytest.mark.parametrize(
    "base_estimator",
    [DecisionTreeClassifier(), ContinuousIntervalTree()],
)
def test_interval_forest_feature_skipping(base_estimator):
    """Test BaseIntervalForest feature skipping with different base estimators."""
    X, y = make_example_3d_numpy(random_state=0)

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
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        interval_features=SevenNumberSummary(),
    )
    est.fit(X, y)

    assert est._efficient_predictions is False


@pytest.mark.parametrize(
    "interval_selection_method",
    ["random", "supervised", "random-supervised"],
)
def test_interval_forest_selection_methods(interval_selection_method):
    """Test BaseIntervalForest with different interval selection methods."""
    X, y = make_example_3d_numpy()

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
    X, y = make_example_3d_numpy(n_timepoints=20)

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=n_intervals,
        series_transformers=[None, FunctionTransformer(np.log1p)],
        random_state=0,
    )
    est.__unit_test_flag = True
    est.fit(X, y)
    est.predict_proba(X)

    assert est._transformed_data[0].shape[1] == n_intervals_len


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
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=0.5,
        interval_features=features,
        replace_nan=0,
        random_state=0,
    )
    est.__unit_test_flag = True
    est.fit(X, y)
    est.predict_proba(X)

    assert est._transformed_data[0].shape[1] == int(output_len * 0.5) * 2


def test_interval_forest_invalid_attribute_subsample():
    """Test BaseIntervalForest with an invalid transformer for subsampling."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=2,
        interval_features=SevenNumberSummary(),
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
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        series_transformers=series_transformer,
        random_state=0,
    )
    est.__unit_test_flag = True
    est.fit(X, y)
    est.predict_proba(X)

    expected = (
        len(series_transformer) * 6 if isinstance(series_transformer, list) else 6
    )
    assert est._transformed_data[0].shape[1] == expected


def test_min_interval_length():
    """Test exception raising for min_interval_length."""
    X, y = make_example_3d_numpy(n_cases=3, n_timepoints=12)

    est = IntervalForestClassifier(min_interval_length=0.5)
    est.fit(X, y)
    assert est._min_interval_length[0] == 6

    est = IntervalForestClassifier(min_interval_length=2.0)
    with pytest.raises(ValueError, match=r"Invalid min_interval_length input"):
        est.fit(X, y)

    series_transformer = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    est = IntervalForestClassifier(
        series_transformers=series_transformer, min_interval_length=[3, 5]
    )
    est.fit(X, y)
    assert est._min_interval_length == [3, 5]

    est = IntervalForestClassifier(
        series_transformers=series_transformer, min_interval_length=[0.5, 0.6]
    )
    est.fit(X, y)
    assert est._min_interval_length == [6, 7]


def test_max_interval_length():
    """Test exception raising max_interval_length."""
    X, y = make_example_3d_numpy(n_cases=3, n_timepoints=12)

    est = IntervalForestClassifier(max_interval_length=0.5)
    est.fit(X, y)
    assert est._max_interval_length[0] == 6

    est = IntervalForestClassifier(max_interval_length=2.0)
    with pytest.raises(ValueError, match=r"Invalid max_interval_length"):
        est.fit(X, y)

    series_transformer = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    est = IntervalForestClassifier(
        series_transformers=series_transformer, max_interval_length=[8, 7]
    )
    est.fit(X, y)
    assert est._max_interval_length == [8, 7]

    est = IntervalForestClassifier(
        series_transformers=series_transformer, max_interval_length=[0.5, 0.7]
    )
    est.fit(X, y)
    assert est._max_interval_length == [6, 8]


def test_interval_features():
    """Testing for the interval_features parameter."""
    X, y = make_example_3d_numpy(n_cases=3, n_timepoints=10)

    f1 = [
        row_mean,
        Catch22(features=["DN_HistogramMode_5", "DN_HistogramMode_10"]),
        row_numba_min,
    ]
    est = IntervalForestClassifier(interval_features=f1)
    est.fit(X, y)

    assert est._interval_function == [True]
    assert est._interval_transformer == [True]
