"""Tests for all series anomaly detectors."""

from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score

from aeon.base._base import _clone_estimator
from aeon.base._base_series import VALID_SERIES_INNER_TYPES
from aeon.testing.testing_data import FULL_TEST_DATA_DICT

# Smallest gap from the constant score baseline that a detector must reach on the
# labelled testing fixture. The fixture is short and holds a single labelled
# anomaly, so this floor is deliberately low. It is here to catch a detector which
# does not discriminate at all, not to grade how good a detector is.
MIN_DISCRIMINATION_MARGIN = 0.05


def _yield_series_anomaly_detection_checks(
    estimator_class, estimator_instances, datatypes
):
    """Yield all anomaly detection checks for an aeon anomaly detector."""
    # only class required
    yield partial(
        check_series_anomaly_detector_overrides_and_tags,
        estimator_class=estimator_class,
    )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # binary output is a label rather than a ranking, so there is no score to
        # rank the time points by. estimator is None when a soft dependency is
        # missing and the instance could not be created
        if (
            estimator is not None
            and estimator.get_tag("anomaly_output_type") == "anomaly_scores"
        ):
            yield partial(
                check_series_anomaly_detector_discriminates,
                estimator=estimator,
                datatype=datatypes[i][0],
            )

        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_series_anomaly_detector_output,
                estimator=estimator,
                datatype=datatype,
            )


def check_series_anomaly_detector_overrides_and_tags(estimator_class):
    """Test compliance with the anomaly detector base class contract."""
    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    if isinstance(X_inner_type, str):
        assert X_inner_type in VALID_SERIES_INNER_TYPES
    else:  # must be a list
        assert all([t in VALID_SERIES_INNER_TYPES for t in X_inner_type])


def check_series_anomaly_detector_output(estimator, datatype):
    """Test the series anomaly detector output on valid data."""
    estimator = _clone_estimator(estimator)

    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
    assert isinstance(y_pred, np.ndarray)
    # series need n_timepoints predictions
    assert len(y_pred) == FULL_TEST_DATA_DICT[datatype]["test"][0].shape[1]

    out_type = estimator.get_tag("anomaly_output_type")
    if out_type == "anomaly_scores":
        assert np.issubdtype(y_pred.dtype, np.floating) or np.issubdtype(
            y_pred.dtype, np.integer
        ), "y_pred must be of floating point or int type"
        assert not np.array_equal(
            np.unique(y_pred), [0, 1]
        ), "y_pred cannot contain only 0s and 1s"
    elif out_type == "binary":
        assert np.issubdtype(y_pred.dtype, np.integer) or np.issubdtype(
            y_pred.dtype, np.bool_
        ), "y_pred must be of int or bool type for binary output"
        assert all(
            val in [0, 1] for val in np.unique(y_pred)
        ), "y_pred must contain only 0s, 1s, True, or False"
    else:
        raise ValueError(f"Unknown anomaly output type: {out_type}")


def check_series_anomaly_detector_discriminates(estimator, datatype):
    """Test the series anomaly detector separates the labelled anomalies.

    The current output check only looks at the shape and dtype of the scores. A
    detector which returns the same score at every time point, or scores unrelated
    to the input, passes it. This check adds the missing part, that the scores
    actually separate the labelled anomalies from the normal time points.

    A constant anomaly score is not wrong in general. On a constant input series a
    constant score is the right answer. This check never sees such a series. It only
    runs on the labelled testing fixture, which does contain anomalies, so a
    constant score on that fixture is a real failure.

    The measure is the area under the ROC curve, which is rank based, so no
    threshold or scale is assumed. Detectors which score anomalies low rather than
    high are accepted, only the distance from the baseline is used.
    """
    estimator = _clone_estimator(estimator, random_state=0)

    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
    y_true = np.asarray(FULL_TEST_DATA_DICT[datatype]["test"][1]).astype(bool)

    # the fixture needs both anomalous and normal time points for the area under
    # the curve to be defined
    if not y_true.any() or y_true.all():
        return

    auc = roc_auc_score(y_true, y_pred)
    # a detector returning one score everywhere lands exactly on chance level, so
    # this reads the baseline off the fixture instead of hard coding it
    baseline = roc_auc_score(y_true, np.zeros(len(y_true)))

    assert abs(auc - baseline) >= MIN_DISCRIMINATION_MARGIN, (
        f"Anomaly scores do not separate the labelled anomalies in the testing "
        f"fixture. Area under the ROC curve is {auc:.3f} against a constant score "
        f"baseline of {baseline:.3f}, a difference of at least "
        f"{MIN_DISCRIMINATION_MARGIN} is required. A single repeated score, or "
        f"scores unrelated to the input, will produce this."
    )
