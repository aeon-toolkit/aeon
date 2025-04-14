"""Tests for all anomaly detectors."""

from functools import partial

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.base._base_series import VALID_SERIES_INNER_TYPES
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)


def _yield_anomaly_detection_checks(estimator_class, estimator_instances, datatypes):
    """Yield all anomaly detection checks for an aeon anomaly detector."""
    # only class required
    yield partial(
        check_anomaly_detector_overrides_and_tags, estimator_class=estimator_class
    )

    # test class instances
    for _, estimator in enumerate(estimator_instances):
        # no data needed
        yield partial(check_anomaly_detector_univariate, estimator=estimator)
        yield partial(check_anomaly_detector_multivariate, estimator=estimator)


def check_anomaly_detector_overrides_and_tags(estimator_class):
    """Test compliance with the anomaly detector base class contract."""
    # Test they don't override final methods, because Python does not enforce this
    assert "fit" not in estimator_class.__dict__
    assert "predict" not in estimator_class.__dict__
    assert "fit_predict" not in estimator_class.__dict__

    # Test that all anomaly detectors implement abstract predict.
    assert "_predict" in estimator_class.__dict__

    # axis class parameter is for internal use only
    assert "axis" not in estimator_class.__dict__

    # Test that fit_is_empty is correctly set
    fit_is_empty = estimator_class.get_class_tag(tag_name="fit_is_empty")
    assert not fit_is_empty == "_fit" not in estimator_class.__dict__

    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    assert X_inner_type in VALID_SERIES_INNER_TYPES

    # Must have at least one set to True
    multi = estimator_class.get_class_tag(tag_name="capability:multivariate")
    uni = estimator_class.get_class_tag(tag_name="capability:univariate")
    assert multi or uni


labels = np.zeros(15, dtype=np.int_)
labels[np.random.choice(15, 5)] = 1
uv_series = make_example_1d_numpy(n_timepoints=15, random_state=0)
uv_series[labels == 1] += 1
mv_series = make_example_2d_numpy_series(n_timepoints=15, n_channels=2, random_state=0)
mv_series[:, labels == 1] += 1


def check_anomaly_detector_univariate(estimator):
    """Test the anomaly detector on univariate data."""
    import pytest

    estimator = _clone_estimator(estimator)

    if estimator.get_tag(tag_name="capability:univariate"):
        pred = estimator.fit_predict(uv_series, labels)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (15,)
        assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))
    else:
        with pytest.raises(ValueError, match="Univariate data not supported"):
            estimator.fit_predict(uv_series, labels)


def check_anomaly_detector_multivariate(estimator):
    """Test the anomaly detector on multivariate data."""
    import pytest

    estimator = _clone_estimator(estimator)

    if estimator.get_tag(tag_name="capability:multivariate"):
        pred = estimator.fit_predict(mv_series, labels)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (15,)
        assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))
    else:
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            estimator.fit_predict(mv_series, labels)
