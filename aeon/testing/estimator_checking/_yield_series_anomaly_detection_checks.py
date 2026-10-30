"""Tests for all series anomaly detectors."""

from functools import partial

from aeon.base._base_series import VALID_SERIES_INNER_TYPES


def _yield_series_anomaly_detection_checks(
    estimator_class, estimator_instances, datatypes
):
    """Yield all anomaly detection checks for an aeon anomaly detector."""
    # only class required
    yield partial(
        check_series_anomaly_detector_overrides_and_tags,
        estimator_class=estimator_class,
    )


def check_series_anomaly_detector_overrides_and_tags(estimator_class):
    """Test compliance with the anomaly detector base class contract."""
    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    if isinstance(X_inner_type, str):
        assert X_inner_type in VALID_SERIES_INNER_TYPES
    else:  # must be a list
        assert all([t in VALID_SERIES_INNER_TYPES for t in X_inner_type])
