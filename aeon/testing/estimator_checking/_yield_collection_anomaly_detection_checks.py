"""Tests for all collection anomaly detectors."""

from functools import partial

from aeon.base._base import _clone_estimator
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import _assert_predict_labels
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES


def _yield_collection_anomaly_detection_checks(
    estimator_class, estimator_instances, datatypes
):
    """Yield all collection anomaly detection checks for an aeon estimator."""
    # only class required
    yield partial(
        check_collection_detector_overrides_and_tags, estimator_class=estimator_class
    )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_collection_detector_output, estimator=estimator, datatype=datatype
            )


def check_collection_detector_overrides_and_tags(estimator_class):
    """Test compliance with the detector base class contract."""
    # Test they don't override final methods, because Python does not enforce this
    final_methods = [
        "fit",
        "predict",
    ]
    for method in final_methods:
        if method in estimator_class.__dict__:
            raise ValueError(
                f"Collection anomaly detector {estimator_class} overrides the "
                f"method {method}. Override _{method} instead."
            )

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


def check_collection_detector_output(estimator, datatype):
    """Test detector outputs the correct data types and values."""
    estimator = _clone_estimator(estimator)

    # run fit and predict
    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
    _assert_predict_labels(y_pred, datatype, unique_labels=[0, 1])
