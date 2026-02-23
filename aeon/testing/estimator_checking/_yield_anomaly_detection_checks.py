"""Tests for all anomaly detectors."""

from functools import partial

from aeon.utils.data_types import ALL_TIME_SERIES_TYPES


def _yield_anomaly_detection_checks(estimator_class, estimator_instances, datatypes):
    """Yield all anomaly detection checks for an aeon anomaly detector."""
    # only class required
    yield partial(
        check_anomaly_detector_overrides_and_tags, estimator_class=estimator_class
    )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # data type irrelevant
        yield partial(
            check_anomaly_detector_learning_types,
            estimator=estimator,
            datatype=datatypes[i][0],
        )


def check_anomaly_detector_overrides_and_tags(estimator_class):
    """Test compliance with the anomaly detector base class contract."""
    # Test they don't override final methods, because Python does not enforce this
    final_methods = ["fit", "predict", "fit_predict"]
    for method in final_methods:
        if method in estimator_class.__dict__:
            raise ValueError(
                f"Anomaly detector {estimator_class} overrides the "
                f"method {method}. Override _{method} instead."
            )

    # Test that all anomaly detectors implement abstract predict.
    assert "_predict" in estimator_class.__dict__

    # Test that fit_is_empty is correctly set
    fit_is_empty = estimator_class.get_class_tag(tag_name="fit_is_empty")
    assert fit_is_empty == ("_fit" not in estimator_class.__dict__)

    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    if isinstance(X_inner_type, str):
        assert X_inner_type in ALL_TIME_SERIES_TYPES
    else:  # must be a list
        assert all([t in ALL_TIME_SERIES_TYPES for t in X_inner_type])

    # Must have at least one set to True
    multi = estimator_class.get_class_tag(tag_name="capability:multivariate")
    uni = estimator_class.get_class_tag(tag_name="capability:univariate")
    assert multi or uni


def check_anomaly_detector_learning_types(estimator, datatype):
    """Test anomaly detector learning types."""
    unsupervised = estimator.get_tag("learning_type:unsupervised")
    semisup = estimator.get_tag("learning_type:semi_supervised")
    supervised = estimator.get_tag("learning_type:supervised")

    assert (
        unsupervised or semisup or supervised
    ), "At least one learning type must be True"
