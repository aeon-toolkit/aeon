"""Tests for the BaseIntervalForest class."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from aeon.base._base import _clone_estimator
from aeon.classification.interval_based._interval_forest import IntervalForestClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.datasets import load_italy_power_demand
from aeon.regression.interval_based._interval_forest import IntervalForestRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import AutocorrelationFunctionTransformer
from aeon.transformations.collection.feature_based import Catch22, SevenNumberSummary
from aeon.utils.numba.stats import row_mean, row_numba_min, row_std


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


@pytest.mark.parametrize(
    "base_estimator",
    [DecisionTreeClassifier(), ContinuousIntervalTree()],
)
def test_temporal_importance_curves(base_estimator):
    """Test temporal_importance_curves for supported base estimators."""
    X, y = load_italy_power_demand(split="train")

    est = IntervalForestClassifier(base_estimator=base_estimator)
    est.fit(X, y)

    names, curves = est.temporal_importance_curves()

    assert isinstance(names, list)
    assert isinstance(curves, list)
    assert len(names) == len(curves) > 0
    assert isinstance(curves[0], np.ndarray)
    assert curves[0].ndim == 1
    assert len(curves[0]) == X.shape[2]
    assert all(np.all(c >= 0) for c in curves)


def test_interval_forest_invalid_base_estimator():
    """Test BaseIntervalForest raises ValueError for invalid base_estimator."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(n_estimators=2, n_intervals=2, base_estimator="bad")
    with pytest.raises(ValueError, match="base_estimator must be a scikit-learn"):
        est.fit(X, y)


def test_interval_forest_invalid_series_transformers():
    """Test BaseIntervalForest raises ValueError for invalid series_transformers."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2, n_intervals=2, series_transformers="invalid"
    )
    with pytest.raises(ValueError, match="Invalid series_transformers input"):
        est.fit(X, y)

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        series_transformers=[None, "invalid"],
    )
    with pytest.raises(ValueError, match="Invalid series_transformers list input"):
        est.fit(X, y)


def test_interval_forest_n_intervals_invalid():
    """Test BaseIntervalForest raises ValueError for invalid n_intervals."""
    X, y = make_example_3d_numpy(n_timepoints=20)
    two_transformers = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    # nested list wrong length
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        n_intervals=[[1, 2], [2], [1]],
    )
    with pytest.raises(ValueError, match="n_intervals as a list or tuple"):
        est.fit(X, y)

    # invalid type (float)
    est = IntervalForestClassifier(n_estimators=2, n_intervals=1.5)
    with pytest.raises(ValueError, match="Invalid n_intervals input"):
        est.fit(X, y)

    # invalid str
    est = IntervalForestClassifier(n_estimators=2, n_intervals="bad_str")
    with pytest.raises(ValueError, match="Invalid str input for n_intervals"):
        est.fit(X, y)

    # nested list with non-int/str inner item
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        n_intervals=[[1, 2], [1.5]],
    )
    with pytest.raises(ValueError):
        est.fit(X, y)

    # outer list with non-int/str/list/tuple item
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        n_intervals=[[1, 2], 1.5],
    )
    with pytest.raises(ValueError):
        est.fit(X, y)


def test_interval_forest_min_interval_length_invalid():
    """Test BaseIntervalForest raises ValueError for invalid min_interval_length."""
    X, y = make_example_3d_numpy(n_cases=3, n_timepoints=12)
    two_transformers = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    est = IntervalForestClassifier(
        series_transformers=two_transformers, min_interval_length=[3, 5, 7]
    )
    with pytest.raises(ValueError, match="min_interval_length as a list"):
        est.fit(X, y)

    est = IntervalForestClassifier(
        series_transformers=two_transformers, min_interval_length=[3, 2.0]
    )
    with pytest.raises(ValueError, match="min_interval_length list items"):
        est.fit(X, y)

    est = IntervalForestClassifier(min_interval_length="invalid")
    with pytest.raises(ValueError, match="Invalid min_interval_length input"):
        est.fit(X, y)


def test_interval_forest_max_interval_length_invalid():
    """Test BaseIntervalForest raises ValueError for invalid max_interval_length."""
    X, y = make_example_3d_numpy(n_cases=3, n_timepoints=12)
    two_transformers = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    est = IntervalForestClassifier(
        series_transformers=two_transformers, max_interval_length=[8, 7, 6]
    )
    with pytest.raises(ValueError, match="max_interval_length as a list"):
        est.fit(X, y)

    est = IntervalForestClassifier(
        series_transformers=two_transformers, max_interval_length=[8, 2.0]
    )
    with pytest.raises(ValueError, match="max_interval_length list items"):
        est.fit(X, y)

    est = IntervalForestClassifier(max_interval_length="invalid")
    with pytest.raises(ValueError, match="Invalid max_interval_length"):
        est.fit(X, y)


def test_interval_forest_single_callable_feature():
    """Test BaseIntervalForest with a single callable as interval_features."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2, n_intervals=2, interval_features=row_mean, random_state=0
    )
    est.fit(X, y)
    assert est.predict_proba(X).shape == (10, 2)


def test_interval_forest_interval_features_nested_list():
    """Test BaseIntervalForest with a nested list for interval_features."""
    X, y = make_example_3d_numpy()
    two_transformers = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    # nested list of callables
    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        series_transformers=two_transformers,
        interval_features=[[row_mean, row_std], [row_mean]],
        random_state=0,
    )
    est.fit(X, y)
    assert est.predict_proba(X).shape == (10, 2)

    # outer list with mixed single callable and nested list
    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        series_transformers=two_transformers,
        interval_features=[row_mean, [row_std]],
        random_state=0,
    )
    est.fit(X, y)
    assert est.predict_proba(X).shape == (10, 2)

    # outer list with single transformer and nested callable list
    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        series_transformers=two_transformers,
        interval_features=[Catch22(features=["DN_HistogramMode_5"]), [row_mean]],
        random_state=0,
    )
    est.fit(X, y)
    assert est.predict_proba(X).shape == (10, 2)


def test_interval_forest_interval_features_invalid():
    """Test BaseIntervalForest raises ValueError for invalid interval_features."""
    X, y = make_example_3d_numpy()
    two_transformers = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    # nested list wrong length
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        interval_features=[[row_mean], [row_mean], [row_mean]],
    )
    with pytest.raises(ValueError, match="interval_features as a list or tuple"):
        est.fit(X, y)

    # invalid type
    est = IntervalForestClassifier(n_estimators=2, interval_features="invalid")
    with pytest.raises(ValueError, match="Invalid interval_features input"):
        est.fit(X, y)

    # invalid item in outer list (not transformer or callable)
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        interval_features=[row_mean, 123],
    )
    with pytest.raises(ValueError):
        est.fit(X, y)

    # invalid item inside nested list
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        interval_features=[[row_mean, 123], [row_mean]],
    )
    with pytest.raises(ValueError):
        est.fit(X, y)


def test_interval_forest_att_subsample_invalid():
    """Test BaseIntervalForest raises ValueError for invalid att_subsample_size."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(n_estimators=2, att_subsample_size=0)
    with pytest.raises(ValueError, match="att_subsample_size must be at least one"):
        est.fit(X, y)

    est = IntervalForestClassifier(n_estimators=2, att_subsample_size=1.5)
    with pytest.raises(ValueError, match="att_subsample_size must be between 0 and 1"):
        est.fit(X, y)

    est = IntervalForestClassifier(n_estimators=2, att_subsample_size=0.0)
    with pytest.raises(ValueError, match="att_subsample_size must be between 0 and 1"):
        est.fit(X, y)

    est = IntervalForestClassifier(n_estimators=2, att_subsample_size="bad")
    with pytest.raises(ValueError, match="Invalid interval_features input"):
        est.fit(X, y)


def test_interval_forest_att_subsample_list():
    """Test BaseIntervalForest with att_subsample_size as a list."""
    X, y = make_example_3d_numpy()
    two_transformers = [FunctionTransformer(np.log1p), FunctionTransformer(np.log1p)]

    # wrong length
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        att_subsample_size=[2, 3, 4],
    )
    with pytest.raises(ValueError, match="att_subsample_size as a list"):
        est.fit(X, y)

    # int < 1 in list
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        att_subsample_size=[2, 0],
    )
    with pytest.raises(ValueError, match="att_subsample_size in list must be at least"):
        est.fit(X, y)

    # float > 1 in list
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        att_subsample_size=[0.5, 1.5],
    )
    with pytest.raises(ValueError, match="att_subsample_size in list must be between"):
        est.fit(X, y)

    # invalid type in list
    est = IntervalForestClassifier(
        n_estimators=2,
        series_transformers=two_transformers,
        att_subsample_size=[0.5, "bad"],
    )
    with pytest.raises(ValueError):
        est.fit(X, y)

    # valid list with None
    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        series_transformers=two_transformers,
        att_subsample_size=[None, 2],
        random_state=0,
    )
    est.fit(X, y)
    assert est.predict_proba(X).shape == (10, 2)


def test_interval_forest_att_subsample_larger_than_features():
    """Test BaseIntervalForest warns when att_subsample_size > number of features."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=100,
        random_state=0,
    )
    with pytest.warns(UserWarning):
        est.fit(X, y)
    assert est.predict_proba(X).shape == (10, 2)


def test_interval_forest_invalid_interval_selection_method():
    """Test BaseIntervalForest raises ValueError for invalid selection."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(n_estimators=2, interval_selection_method=123)
    with pytest.raises(ValueError, match="Unknown interval_selection_method"):
        est.fit(X, y)

    est = IntervalForestClassifier(
        n_estimators=2, interval_selection_method="bad_method"
    )
    with pytest.raises(ValueError, match="Unknown interval_selection_method"):
        est.fit(X, y)


def test_interval_forest_supervised_with_transformer_features():
    """Test supervised selection raises ValueError when transformer features used."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        interval_selection_method="supervised",
        interval_features=Catch22(features=["DN_HistogramMode_5"]),
    )
    with pytest.raises(ValueError, match="Supervised interval_selection_method"):
        est.fit(X, y)


def test_interval_forest_invalid_replace_nan():
    """Test BaseIntervalForest raises ValueError for invalid replace_nan."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(n_estimators=2, replace_nan=[1, 2])
    with pytest.raises(ValueError, match="Invalid replace_nan input"):
        est.fit(X, y)


def test_interval_forest_replace_nan_options():
    """Test BaseIntervalForest with replace_nan string and numeric options."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=2, n_intervals=2, replace_nan="nan", random_state=0
    )
    est.fit(X, y)
    assert est.predict(X).shape == (10,)

    est = IntervalForestClassifier(
        n_estimators=2, n_intervals=2, replace_nan=0, random_state=0
    )
    est.fit(X, y)
    assert est.predict(X).shape == (10,)


def test_interval_forest_time_limit():
    """Test BaseIntervalForest respects time_limit_in_minutes contract."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(
        n_estimators=500,
        n_intervals=2,
        time_limit_in_minutes=0.01,
        contract_max_n_estimators=10,
        random_state=0,
    )
    est.fit(X, y)
    assert est.predict_proba(X).shape == (10, 2)
    assert est._n_estimators <= 10


def test_interval_forest_regressor():
    """Test BaseIntervalForest as a regressor."""
    X, _ = make_example_3d_numpy()
    y = np.random.default_rng(0).random(10)

    est = IntervalForestRegressor(n_estimators=2, n_intervals=2, random_state=0)
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (10,)


def test_interval_forest_regressor_fit_predict():
    """Test BaseIntervalForest fit_predict for regressor."""
    X, _ = make_example_3d_numpy()
    y = np.random.default_rng(0).random(10)

    est = IntervalForestRegressor(n_estimators=2, n_intervals=2, random_state=0)
    preds = est.fit_predict(X, y)
    assert preds.shape == (10,)


def test_interval_forest_classifier_fit_predict():
    """Test BaseIntervalForest fit_predict and fit_predict_proba for classifier."""
    X, y = make_example_3d_numpy()

    est = IntervalForestClassifier(n_estimators=2, n_intervals=2, random_state=0)
    preds = est.fit_predict(X, y)
    assert preds.shape == (10,)

    est = IntervalForestClassifier(n_estimators=2, n_intervals=2, random_state=0)
    probas = est.fit_predict_proba(X, y)
    assert probas.shape == (10, 2)


def test_temporal_importance_curves_options():
    """Test temporal_importance_curves with return_dict and normalise_time_points."""
    X, y = load_italy_power_demand(split="train")

    est = IntervalForestClassifier(random_state=0)
    est.fit(X, y)

    curves_dict = est.temporal_importance_curves(return_dict=True)
    assert isinstance(curves_dict, dict)

    names, curves = est.temporal_importance_curves(normalise_time_points=True)
    assert isinstance(names, list)
    assert isinstance(curves, list)


def test_temporal_importance_curves_errors():
    """Test temporal_importance_curves raises for unsupported configurations."""
    X, _ = make_example_3d_numpy()
    y_reg = np.random.default_rng(0).random(10)
    _, y_cls = make_example_3d_numpy()

    reg = IntervalForestRegressor(n_estimators=2, n_intervals=2, random_state=0)
    reg.fit(X, y_reg)
    with pytest.raises(NotImplementedError):
        reg.temporal_importance_curves()

    est = IntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        base_estimator=RandomForestClassifier(n_estimators=2),
        random_state=0,
    )
    est.fit(X, y_cls)
    with pytest.raises(ValueError, match="base_estimator for temporal importance"):
        est.temporal_importance_curves()
