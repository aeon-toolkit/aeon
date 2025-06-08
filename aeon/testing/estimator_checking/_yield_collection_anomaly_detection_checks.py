"""Tests for all collection anomaly detectors."""

from functools import partial

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation import get_n_cases


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
                check_collection_anomaly_detector_output,
                estimator=estimator,
                datatype=datatype,
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


def check_collection_anomaly_detector_output(estimator, datatype):
    """Test the collection anomaly detector output on valid data."""
    estimator = _clone_estimator(estimator)

    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
    assert isinstance(y_pred, np.ndarray)
    # collections need n_cases predictions
    assert len(y_pred) == get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0])

    ot = estimator.get_tag("anomaly_output_type")
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
