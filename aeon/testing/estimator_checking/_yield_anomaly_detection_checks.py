"""Tests for all anomaly detectors."""

from functools import partial

import numpy as np

from aeon.anomaly_detection.collection.base import BaseCollectionAnomalyDetector
from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.base._base import _clone_estimator
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.deep_equals import deep_equals
from aeon.utils.data_types import ALL_TIME_SERIES_TYPES
from aeon.utils.validation.collection import get_n_cases


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

        methods = ["predict"]
        # Collection checks already covered the inherited fit_predict implementation.
        if "_fit_predict" in estimator_class.__dict__ or issubclass(
            estimator_class, BaseCollectionAnomalyDetector
        ):
            methods.append("fit_predict")

        for datatype in datatypes[i]:
            for method in methods:
                yield partial(
                    check_anomaly_detector_output,
                    estimator=estimator,
                    datatype=datatype,
                    method=method,
                )
            if "_fit_predict" in estimator_class.__dict__:
                yield partial(
                    check_anomaly_detector_fit_predict_equality,
                    estimator=estimator,
                    datatype=datatype,
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

    assert unsupervised or semisup or supervised, (
        "At least one learning type must be True"
    )


def check_anomaly_detector_output(estimator, datatype, method):
    """Test anomaly detector predict and fit_predict output on valid data."""
    estimator = _clone_estimator(estimator)
    X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]

    if method == "fit_predict":
        X = X_train
        y_pred = estimator.fit_predict(X, y_train)
    else:
        estimator.fit(X_train, y_train)
        X = FULL_TEST_DATA_DICT[datatype]["test"][0]
        y_pred = estimator.predict(X)

    assert isinstance(y_pred, np.ndarray)
    if isinstance(estimator, BaseSeriesAnomalyDetector):
        assert len(y_pred) == X.shape[1]
    else:
        assert len(y_pred) == get_n_cases(X)

    out_type = estimator.get_tag("anomaly_output_type")
    if out_type == "anomaly_scores":
        assert np.issubdtype(y_pred.dtype, np.floating) or np.issubdtype(
            y_pred.dtype, np.integer
        ), "y_pred must be of floating point or int type"
        assert not np.array_equal(np.unique(y_pred), [0, 1]), (
            "y_pred cannot contain only 0s and 1s"
        )
    elif out_type == "binary":
        assert np.issubdtype(y_pred.dtype, np.integer) or np.issubdtype(
            y_pred.dtype, np.bool_
        ), "y_pred must be of int or bool type for binary output"
        assert all(val in [0, 1] for val in np.unique(y_pred)), (
            "y_pred must contain only 0s, 1s, True, or False"
        )
    else:
        raise ValueError(f"Unknown anomaly output type: {out_type}")


def check_anomaly_detector_fit_predict_equality(estimator, datatype):
    """Compare fit_predict with fit followed by predict on the same data."""
    estimator_fit_predict = _clone_estimator(estimator, random_state=42)
    estimator = _clone_estimator(estimator, random_state=42)
    X, y = FULL_TEST_DATA_DICT[datatype]["train"]

    y_pred_fit_predict = estimator_fit_predict.fit_predict(X, y)
    y_pred = estimator.fit(X, y).predict(X)

    equal, msg = deep_equals(y_pred_fit_predict, y_pred, return_msg=True)
    assert equal, msg
