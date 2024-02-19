"""Tests for Mock Forecasters."""

__maintainer__ = []

from copy import deepcopy

import pytest
from pandas.testing import assert_series_equal

from aeon.classification.base import BaseClassifier
from aeon.clustering.base import BaseClusterer
from aeon.datasets import load_airline
from aeon.forecasting.base import BaseForecaster, ForecastingHorizon
from aeon.forecasting.naive import NaiveForecaster
from aeon.testing.mock_estimators import (
    MockUnivariateForecasterLogger,
    make_mock_estimator,
)
from aeon.testing.mock_estimators._mock_forecasters import (
    _method_logger,
    _MockEstimatorMixin,
)
from aeon.testing.utils.deep_equals import deep_equals
from aeon.transformations.base import BaseTransformer
from aeon.transformations.boxcox import BoxCoxTransformer

y_series = load_airline().iloc[:-5]
y_frame = y_series.to_frame()
X_series_train = load_airline().iloc[:-5]
X_series_pred = load_airline().iloc[-5:]
X_frame_train = X_series_train.to_frame()
X_frame_pred = X_series_pred.to_frame()
fh_absolute = ForecastingHorizon(values=X_series_pred.index, is_relative=False)
fh_relative = ForecastingHorizon(values=[1, 2, 3], is_relative=True)


@pytest.mark.parametrize(
    "base", [BaseForecaster, BaseClassifier, BaseClusterer, BaseTransformer]
)
def test_mixin(base):
    """Test _MockEstimatorMixin is valid for all aeon estimator base classes."""

    class _DummyClass(base, _MockEstimatorMixin):
        def __init__(self):
            super().__init__()

        def _fit(self):
            """Empty method, here for testing purposes."""
            pass

        def _predict(self):
            """Empty method, here for testing purposes."""
            pass

        def _score(self):
            """Empty method, here for testing purposes."""
            pass

    dummy_instance = _DummyClass()
    assert hasattr(dummy_instance, "log")
    dummy_instance.add_log_item(42)
    assert hasattr(dummy_instance, "_MockEstimatorMixin__log")


def test_add_log_item():
    """Test _MockEstimatorMixin.add_log_item behaviour."""
    mixin = _MockEstimatorMixin()
    mixin.add_log_item(1)
    mixin.add_log_item(2)
    assert mixin.log[0] == 1
    assert mixin.log[1] == 2


def test_log_is_property():
    """Test _MockEstimatorMixin.log can't be overwritten."""
    mixin = _MockEstimatorMixin()
    with pytest.raises(AttributeError) as excinfo:
        mixin.log = 1
        assert "can't set attribute" in str(excinfo.value)


def test_method_logger_exception():
    """Test that _method_logger only works for _MockEstimatorMixin subclasses."""

    class _DummyClass:
        def __init__(self) -> None:
            """Empty method, here for testing purposes."""
            pass

        @_method_logger
        def _method(self):
            """Empty method, here for testing purposes."""
            pass

    with pytest.raises(TypeError) as excinfo:
        dummy_instance = _DummyClass()
        dummy_instance._method()
        assert "Estimator is not a Mock Estimator" in str(excinfo.value)


def test_method_logger():
    """Test that method logger returns the correct output."""

    class _DummyClass(_MockEstimatorMixin):
        def __init__(self) -> None:
            super().__init__()

        @_method_logger
        def _method1(self, positional_param, optional_param="test_optional"):
            """Empty method, here for testing purposes."""
            pass

        @_method_logger
        def _method2(self, positional_param, optional_param="test_optional_2"):
            """Empty method, here for testing purposes."""
            pass

        @_method_logger
        def _method3(self):
            """Empty method, here for testing purposes."""
            pass

    dummy_instance = _DummyClass()
    dummy_instance._method1("test_positional")
    assert dummy_instance.log == [
        (
            "_method1",
            {"positional_param": "test_positional", "optional_param": "test_optional"},
        )
    ]
    dummy_instance._method2("test_positional_2")
    assert dummy_instance.log == [
        (
            "_method1",
            {"positional_param": "test_positional", "optional_param": "test_optional"},
        ),
        (
            "_method2",
            {
                "positional_param": "test_positional_2",
                "optional_param": "test_optional_2",
            },
        ),
    ]
    dummy_instance._method3()
    assert dummy_instance.log == [
        (
            "_method1",
            {"positional_param": "test_positional", "optional_param": "test_optional"},
        ),
        (
            "_method2",
            {
                "positional_param": "test_positional_2",
                "optional_param": "test_optional_2",
            },
        ),
        ("_method3", {}),
    ]


@pytest.mark.parametrize(
    "estimator_class, method_regex, logged_methods",
    [
        (NaiveForecaster, r"(?!^_\w+)", ["fit"]),
        (NaiveForecaster, ".*", ["fit", "_fit"]),
        (BoxCoxTransformer, r"(?!^_\w+)", ["fit"]),
        (BoxCoxTransformer, ".*", ["fit", "_fit"]),
    ],
)
def test_make_mock_estimator(estimator_class, method_regex, logged_methods):
    """Test that make_mock_estimator output logs the right methods."""
    estimator = make_mock_estimator(estimator_class, method_regex)()
    estimator.fit(y_series)
    methods_called = [entry[0] for entry in estimator.log]

    assert set(methods_called) >= set(logged_methods)


@pytest.mark.parametrize(
    "estimator_class, estimator_kwargs",
    [
        (NaiveForecaster, {"strategy": "last", "sp": 2, "window_length": None}),
        (NaiveForecaster, {"strategy": "mean", "sp": 1, "window_length": None}),
    ],
)
def test_make_mock_estimator_with_kwargs(estimator_class, estimator_kwargs):
    """Test that make_mock_estimator behaves like the passed estimator."""
    mock_estimator = make_mock_estimator(estimator_class)
    mock_estimator_instance = mock_estimator(estimator_kwargs)
    estimator_instance = estimator_class(**estimator_kwargs)
    mock_estimator_instance.fit(y_series)
    estimator_instance.fit(y_series)

    assert_series_equal(
        estimator_instance.predict(fh=[1, 2, 3]),
        mock_estimator_instance.predict(fh=[1, 2, 3]),
    )
    assert (
        (mock_estimator_instance.strategy == estimator_kwargs["strategy"])
        and (mock_estimator_instance.sp == estimator_kwargs["sp"])
        and (mock_estimator_instance.window_length == estimator_kwargs["window_length"])
    )


@pytest.mark.parametrize(
    "y, X_train, X_pred, fh",
    [
        (y_series, X_series_train, X_series_pred, fh_absolute),
        (y_series, X_frame_train, X_frame_pred, fh_absolute),
        (y_series, None, None, fh_absolute),
        (y_series, None, None, fh_relative),
        (y_frame, None, None, fh_relative),
    ],
)
def test_mock_univariate_forecaster_log(y, X_train, X_pred, fh):
    """Tests the log of the MockUnivariateForecasterLogger.

    Tests the following:
    - log format and content
    - All the private methods that have logging enabled are in the log
    - the correct inner types are preserved, according to the forecaster tags
    """
    forecaster = MockUnivariateForecasterLogger()
    forecaster.fit(y, X_train, fh)
    forecaster.predict(fh, X_pred)
    forecaster.update(y, X_train, fh)
    forecaster.predict_quantiles(fh=fh, X=X_pred, alpha=[0.1, 0.9])

    _X_train = deepcopy(X_frame_train) if X_train is not None else None
    _X_pred = deepcopy(X_frame_pred) if X_pred is not None else None

    expected_log = [
        ("_fit", {"y": y_series, "X": _X_train, "fh": fh}),
        ("_predict", {"fh": fh, "X": _X_pred}),
        ("_update", {"y": y_series, "X": _X_train, "update_params": fh}),
        (
            "_predict_quantiles",
            {"fh": fh, "X": _X_pred, "alpha": [0.1, 0.9]},
        ),
    ]

    assert deep_equals(forecaster.log, expected_log)
