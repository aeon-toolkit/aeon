"""Tests for all forecasters."""

from functools import partial

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.base._base_series import VALID_SERIES_INPUT_TYPES


def _yield_forecasting_checks(estimator_class, estimator_instances, datatypes):
    """Yield all forecasting checks for an aeon forecaster."""
    # only class required
    yield partial(check_forecasting_base_functionality, estimator_class=estimator_class)

    # test class instances
    for _, estimator in enumerate(estimator_instances):
        # no data needed
        yield partial(check_forecaster_instance, estimator=estimator)


def check_forecasting_base_functionality(estimator_class):
    """Test compliance with the base class contract."""
    # Test they dont override final methods, because python does not enforce this
    assert "fit" not in estimator_class.__dict__
    assert "predict" not in estimator_class.__dict__
    assert "forecast" not in estimator_class.__dict__
    fit_is_empty = estimator_class.get_class_tag(tag_name="fit_is_empty")
    assert not fit_is_empty == "_fit" not in estimator_class.__dict__
    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    assert X_inner_type in VALID_SERIES_INPUT_TYPES
    # Must have at least one set to True
    multi = estimator_class.get_class_tag(tag_name="capability:multivariate")
    uni = estimator_class.get_class_tag(tag_name="capability:univariate")
    assert multi or uni


def check_forecaster_instance(estimator):
    """Test forecasters."""
    estimator = _clone_estimator(estimator)
    pass
    # Sort
    # Check output correct: predict should return a float
    y = np.array([0.5, 0.7, 0.8, 0.9, 1.0])
    estimator.fit(y)
    p = estimator.predict()
    assert isinstance(p, float)
    # forecast should return a float equal to fit/predict
    p2 = estimator.forecast(y)
    assert p == p2
