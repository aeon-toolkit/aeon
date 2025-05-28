"""Tests for all anomaly detectors."""

from functools import partial

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.base._base_series import VALID_SERIES_INNER_TYPES
from aeon.testing.testing_data import FULL_TEST_DATA_DICT


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

        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_anomaly_detector_output, estimator=estimator, datatype=datatype
            )


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


def check_anomaly_detector_learning_types(estimator, datatype):
    """Test anomaly detector learning types."""
    unsupervised = estimator.get_tag("learning_type:unsupervised")
    semisup = estimator.get_tag("learning_type:semi_supervised")
    supervised = estimator.get_tag("learning_type:supervised")

    assert (
        unsupervised or semisup or supervised
    ), "At least one learning type must be True"


def check_anomaly_detector_output(estimator, datatype):
    """Test the anomaly detector output on valid data."""
    estimator1 = _clone_estimator(estimator, random_state=42)
    estimator2 = _clone_estimator(estimator, random_state=42)
    estimator_class = type(estimator)

    estimator1.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    y_pred = estimator1.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == FULL_TEST_DATA_DICT[datatype]["test"][0].shape[1]

    ot = estimator1.get_tag("anomaly_output_type")
    if ot == "anomaly_scores":
        assert np.issubdtype(y_pred.dtype, np.floating) or np.issubdtype(
            y_pred.dtype, np.integer
        ), "y_pred must be of floating point or int type"
        assert not np.array_equal(
            np.unique(y_pred), [0, 1]
        ), "y_pred cannot contain only 0s and 1s"
    elif ot == "binary":
        assert np.issubdtype(y_pred.dtype, np.integer) or np.issubdtype(
            y_pred.dtype, np.bool_
        ), "y_pred must be of int or bool type for binary output"
        assert all(
            val in [0, 1] for val in np.unique(y_pred)
        ), "y_pred must contain only 0s, 1s, True, or False"
    else:
        raise ValueError(f"Unknown anomaly output type: {ot}")

    # check _fit_predict output is same as fit().predict()
    if "_fit_predict" not in estimator_class.__dict__:
        return

    expected_output = estimator1.predict(FULL_TEST_DATA_DICT[datatype]["train"][0])

    fit_predict_output = estimator2.fit_predict(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    assert np.array_equal(
        fit_predict_output, expected_output
    ), "outputs of _fit_predict() does not match fit().predict()"
