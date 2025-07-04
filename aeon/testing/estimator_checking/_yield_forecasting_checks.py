"""Tests for all forecasters."""

from functools import partial

from aeon.base._base import _clone_estimator
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.utils.data_types import VALID_SERIES_INNER_TYPES


def _yield_forecasting_checks(estimator_class, estimator_instances, datatypes):
    """Yield all forecasting checks for an aeon forecaster."""
    # only class required
    yield partial(check_forecaster_overrides_and_tags, estimator_class=estimator_class)

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_forecaster_output,
                estimator=estimator,
                datatype=datatype,
            )


def check_forecaster_overrides_and_tags(estimator_class):
    """Test compliance with the forecaster base class contract."""
    # Test they don't override final methods, because Python does not enforce this
    final_methods = ["fit", "predict", "forecast"]
    for method in final_methods:
        if method in estimator_class.__dict__:
            raise ValueError(
                f"Forecaster {estimator_class} overrides the "
                f"method {method}. Override _{method} instead."
            )

    # Test that all forecasters implement abstract predict.
    assert "_predict" in estimator_class.__dict__

    # todo decide what to do with "fit_is_empty" and abstract "_fit"

    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    if isinstance(X_inner_type, str):
        assert X_inner_type in VALID_SERIES_INNER_TYPES
    else:  # must be a list
        assert all([t in VALID_SERIES_INNER_TYPES for t in X_inner_type])

    # Must have at least one set to True
    multi = estimator_class.get_class_tag(tag_name="capability:multivariate")
    uni = estimator_class.get_class_tag(tag_name="capability:univariate")
    assert multi or uni, (
        "At least one of tag capability:multivariate or "
        "capability:univariate must be true."
    )


def check_forecaster_output(estimator, datatype):
    """Test the forecaster output on valid data."""
    estimator = _clone_estimator(estimator)

    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
    )
    assert hasattr(estimator, "forecast_"), "fit() must set the attribute forecast_"
    assert isinstance(estimator.forecast_, float), "attribute forecast_ must be a float"

    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])

    assert isinstance(y_pred, float), (
        f"predict(y) output should be float, got" f" {type(y_pred)}"
    )

    y_pred = estimator.forecast_
    y_pred2 = estimator.forecast(FULL_TEST_DATA_DICT[datatype]["train"][0])
    assert y_pred == y_pred2, (
        f"estimator.forecast_ and forecast(y) output differ: {y_pred} !=" f" {y_pred2}"
    )
