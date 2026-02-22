"""Tests for target label validation functions."""

import numpy as np
import pandas as pd
import pytest

from aeon.utils.validation.labels import (
    check_anomaly_detection_y,
    check_classification_y,
    check_regression_y,
)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 1, 0, 1]),
        np.array(["a", "b", "a", "b"]),
        pd.Series([0, 1, 0, 1]),
        pd.Series(["a", "b", "a", "b"]),
    ],
)
def test_check_classification_y_allows_binary_targets(y):
    """Accept binary targets for classification."""
    check_classification_y(y)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 1, 2, 1, 0]),
        np.array(["a", "b", "c", "b", "a"]),
        pd.Series([0, 1, 2, 1, 0]),
        pd.Series(["a", "b", "c", "b", "a"]),
    ],
)
def test_check_classification_y_allows_multiclass_targets(y):
    """Accept multiclass targets for classification."""
    check_classification_y(y)


@pytest.mark.parametrize(
    "y", [None, 123, "abc", [0, 1], (0, 1), {0, 1}, pd.DataFrame({"a": [0, 1]})]
)
def test_check_classification_y_rejects_non_array_or_series(y):
    """Reject non-numpy/non-Series y inputs for classification."""
    with pytest.raises(TypeError, match=r"y must be a np\.array or a pd\.Series"):
        check_classification_y(y)


def test_check_classification_y_rejects_ndim():
    """Reject multi-dimensional numpy y for classification."""
    y = np.array([[0, 1], [1, 0]])
    with pytest.raises(TypeError, match=r"y must be 1-dimensional, found 2 dimensions"):
        check_classification_y(y)


@pytest.mark.parametrize("y", [np.array([]), pd.Series([], dtype=float)])
def test_check_classification_y_rejects_empty(y):
    """Reject empty y for classification."""
    with pytest.raises(ValueError, match=r"y must not be empty\."):
        check_classification_y(y)


@pytest.mark.parametrize("y", [np.array([0.1, 0.2, 0.3]), pd.Series([0.1, 0.2, 0.3])])
def test_check_classification_y_rejects_continuous_targets(y):
    """Reject continuous targets for classification."""
    with pytest.raises(
        ValueError,
        match=r"y type is .* which is not valid for "
        r"classification\..*binary or multiclass",
    ):
        check_classification_y(y)


@pytest.mark.parametrize("y", [np.array([0, 0, 0]), pd.Series([0, 0, 0])])
def test_check_classification_y_rejects_single_unique_label(y):
    """Reject classification y with fewer than 2 unique labels."""
    with pytest.raises(
        ValueError, match=r"y must contain at least 2 unique labels, but found 1\."
    ):
        check_classification_y(y)


def test_check_classification_y_does_not_mutate_input():
    """Ensure check_classification_y does not mutate numpy or Series y."""
    y_np = np.array([0, 1, 0, 1])
    y_np_before = y_np.copy()
    check_classification_y(y_np)
    assert np.array_equal(y_np, y_np_before)

    y_pd = pd.Series([0, 1, 0, 1])
    y_pd_before = y_pd.copy()
    check_classification_y(y_pd)
    pd.testing.assert_series_equal(y_pd, y_pd_before)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 2.5, 3.25]),
        pd.Series([0.1, 0.2, 0.3]),
        pd.Series([1.0, 2.5, 3.25]),
    ],
)
def test_check_regression_y_allows_continuous_targets(y):
    """Accept continuous targets for regression."""
    check_regression_y(y)


@pytest.mark.parametrize(
    "y",
    [
        None,
        123,
        "abc",
        [0.1, 0.2],
        (0.1, 0.2),
        {0.1, 0.2},
        pd.DataFrame({"a": [0.1, 0.2]}),
    ],
)
def test_check_regression_y_rejects_non_array_or_series(y):
    """Reject non-numpy/non-Series y for regression."""
    with pytest.raises(TypeError, match=r"y must be a np\.array or a pd\.Series"):
        check_regression_y(y)


def test_check_regression_y_rejects_ndim():
    """Reject multi-dimensional y for regression."""
    y = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(TypeError, match=r"y must be 1-dimensional, found 2 dimensions"):
        check_regression_y(y)


@pytest.mark.parametrize("y", [np.array([]), pd.Series([], dtype=float)])
def test_check_regression_y_rejects_empty(y):
    """Reject empty y for regression."""
    with pytest.raises(ValueError, match=r"y must not be empty\."):
        check_regression_y(y)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 2, 1, 0]),
        pd.Series([0, 1, 0, 1]),
        pd.Series([0, 1, 2, 1, 0]),
    ],
)
def test_check_regression_y_rejects_non_continuous_targets(y):
    """Reject non-continuous targets for regression."""
    with pytest.raises(
        ValueError,
        match=r"y type is .* which is not valid for regression\..*continuous",
    ):
        check_regression_y(y)


def test_check_regression_y_does_not_mutate_input():
    """Ensure check_regression_y does not mutate numpy or Series y."""
    y_np = np.array([0.1, 0.2, 0.3])
    y_np_before = y_np.copy()
    check_regression_y(y_np)
    assert np.array_equal(y_np, y_np_before)

    y_pd = pd.Series([0.1, 0.2, 0.3])
    y_pd_before = y_pd.copy()
    check_regression_y(y_pd)
    pd.testing.assert_series_equal(y_pd, y_pd_before)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 1, 0, 1]),
        np.array([False, True, False, True]),
        pd.Series([0, 1, 0, 1]),
        pd.Series([False, True, False, True]),
    ],
)
def test_check_anomaly_detection_y_allows_binary_targets(y):
    """Accept 0/1 targets for anomaly detection."""
    check_anomaly_detection_y(y)


@pytest.mark.parametrize(
    "y", [None, 123, "abc", [0, 1], (0, 1), {0, 1}, pd.DataFrame({"a": [0, 1]})]
)
def test_check_anomaly_detection_y_rejects_non_array_or_series(y):
    """Reject non-numpy/non-Series y for anomaly detection."""
    with pytest.raises(TypeError, match=r"y must be a np\.array or a pd\.Series"):
        check_anomaly_detection_y(y)


def test_check_anomaly_detection_y_rejects_ndim():
    """Reject multi-dimensional y for anomaly detection."""
    y = np.array([[0, 1], [1, 0]])
    with pytest.raises(TypeError, match=r"y must be 1-dimensional, found 2 dimensions"):
        check_anomaly_detection_y(y)


@pytest.mark.parametrize("y", [np.array([]), pd.Series([], dtype=int)])
def test_check_anomaly_detection_y_rejects_empty(y):
    """Reject empty y for anomaly detection with a clear ValueError."""
    with pytest.raises(ValueError, match=r"y must not be empty\."):
        check_anomaly_detection_y(y)


@pytest.mark.parametrize("y", [np.array([0, 0, 0]), pd.Series([0, 0, 0])])
def test_check_anomaly_detection_y_rejects_single_label(y):
    """Reject anomaly y with fewer than 2 unique labels."""
    with pytest.raises(
        ValueError, match=r"y must contain at least 2 unique labels, but found 1\."
    ):
        check_anomaly_detection_y(y)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 2, 0, 1]),
        pd.Series([0, 2, 0, 1]),
        np.array([0.1, 0.2, 0.3]),
        pd.Series([0.1, 0.2, 0.3]),
        np.array(["a", "b", "a", "b"]),
        pd.Series(["a", "b", "a", "b"]),
        np.array([0, 0, 1, 1, np.nan]),
        pd.Series([0, 0, 1, 1, pd.NA]),
    ],
)
def test_check_anomaly_detection_y_rejects_non_binary(y):
    """Reject anomaly y that contains values other than 0/1."""
    with pytest.raises(
        ValueError,
        match=r"y input must only contain 0 \(not anomalous\) "
        r"or 1 \(anomalous\) values\.",
    ):
        check_anomaly_detection_y(y)


def test_check_anomaly_detection_y_does_not_mutate_input():
    """Ensure check_anomaly_detection_y does not mutate numpy or Series y."""
    y_np = np.array([0, 1, 0, 1])
    y_np_before = y_np.copy()
    check_anomaly_detection_y(y_np)
    assert np.array_equal(y_np, y_np_before)

    y_pd = pd.Series([0, 1, 0, 1])
    y_pd_before = y_pd.copy()
    check_anomaly_detection_y(y_pd)
    pd.testing.assert_series_equal(y_pd, y_pd_before)
