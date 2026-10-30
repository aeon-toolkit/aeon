"""Tests for all collection anomaly detectors."""

from functools import partial

from aeon.utils.data_types import COLLECTIONS_DATA_TYPES


def _yield_collection_anomaly_detection_checks(
    estimator_class, estimator_instances, datatypes
):
    """Yield all collection anomaly detection checks for an aeon estimator."""
    # only class required
    yield partial(
        check_collection_detector_overrides_and_tags, estimator_class=estimator_class
    )


def check_collection_detector_overrides_and_tags(estimator_class):
    """Test compliance with the detector base class contract."""
    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    if isinstance(X_inner_type, str):
        assert X_inner_type in COLLECTIONS_DATA_TYPES
    else:  # must be a list
        assert all([t in COLLECTIONS_DATA_TYPES for t in X_inner_type])

    # one of X_inner_types must be capable of storing unequal length
    if estimator_class.get_class_tag("capability:unequal_length"):
        valid_unequal_types = ["np-list", "df-list", "pd-multiindex"]
        if isinstance(X_inner_type, str):
            assert X_inner_type in valid_unequal_types
        else:  # must be a list
            assert any([t in valid_unequal_types for t in X_inner_type])
