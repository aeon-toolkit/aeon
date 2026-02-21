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
    "y",
    [
        None,
        123,
        "abc",
        [0, 1, 0, 1],
        (0, 1, 0, 1),
        {0, 1},
        np.array([0.1, 0.2, 0.3]),
        pd.Series([0.1, 0.2, 0.3]),
        np.array([[0, 1], [1, 0]]),
        pd.DataFrame({"a": [0, 1], "b": [1, 0]}),
        np.array([["a"], ["b"], ["c"]]),
        np.array([0, 0, 0]),
        pd.Series([0, 0, 0]),
    ],
)
def test_check_classification_y_rejects_non_binary_or_multiclass_targets(y):
    """Reject y that is not binary or multiclass for classification."""
    if isinstance(y, pd.DataFrame) or not isinstance(y, (pd.Series, np.ndarray)):
        with pytest.raises(TypeError, match=r"y must be a np.array or a pd.Series"):
            check_classification_y(y)
    elif isinstance(y, np.ndarray) and y.ndim > 1:
        with pytest.raises(TypeError, match=r"y must be 1-dimensional"):
            check_classification_y(y)
    else:
        with pytest.raises(ValueError, match=r"not valid for classification"):
            check_classification_y(y)


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
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 2, 1, 0]),
        pd.Series([0, 1, 0, 1]),
        pd.Series([0, 1, 2, 1, 0]),
        np.array([[0], [1], [0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([0, 0, 0]),
        pd.Series([0, 0, 0]),
    ],
)
def test_check_regression_y_rejects_non_continuous_targets(y):
    """Reject y that is not continuous for regression."""
    if isinstance(y, np.ndarray) and y.ndim > 1:
        with pytest.raises(TypeError, match=r"y must be 1-dimensional"):
            check_regression_y(y)
    else:
        with pytest.raises(ValueError, match=r"not valid for regression"):
            check_regression_y(y)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 1, 0, 1]),
        np.array([False, True, False, True]),
        pd.Series([0, 1, 0, 1]),
        pd.Series([False, True, False, True]),
        np.array([[0, 1], [1, 0]]),
    ],
)
def test_check_anomaly_detection_y_allows_binary_targets(y):
    """Accept 0/1 targets for anomaly detection."""
    check_anomaly_detection_y(y)


@pytest.mark.parametrize(
    "y",
    [
        None,
        123,
        "abc",
        [0, 1],
        (0, 1),
        {0, 1},
        np.array([0, 1, 2, 1, 0]),
        pd.Series([0, 1, 2, 1, 0]),
        np.array([0.1, 0.2, 0.3]),
        pd.Series([0.1, 0.2, 0.3]),
        np.array([[0], [1], [0]]),
        np.array([[0, 1], [1, 0]]),
        np.array(["a", "b", "a", "b"]),
        pd.Series(["a", "b", "a", "b"]),
        np.array([0, 0, 0]),
        pd.Series([0, 0, 0]),
    ],
)
def test_check_anomaly_detection_y_rejects_non_binary_targets(y):
    """Reject y that is not 0/1 for anomaly detection."""
    if isinstance(y, np.ndarray) and y.ndim > 1:
        with pytest.raises(TypeError, match=r"y must be 1-dimensional"):
            check_anomaly_detection_y(y)
    else:
        with pytest.raises(ValueError, match=r"not valid for anomaly detection"):
            check_anomaly_detection_y(y)


def test_check_functions_do_not_mutate():
    """Ensure label checkers do not mutate the provided y."""
    y_np = np.array([0, 1, 0, 1])
    y_pd = pd.Series([0.1, 0.2, 0.3])

    y_np_before = y_np.copy()
    y_pd_before = y_pd.copy()

    check_classification_y(y_np)
    check_anomaly_detection_y(y_np)
    check_regression_y(y_pd)

    assert np.array_equal(y_np, y_np_before)
    pd.testing.assert_series_equal(y_pd, y_pd_before)


def test_empty_y_is_rejected():
    """Lock in behaviour for empty y (should error rather than silently pass)."""
    y_np = np.array([])
    y_pd = pd.Series([], dtype=float)

    for fn in (check_classification_y, check_regression_y, check_anomaly_detection_y):
        with pytest.raises(ValueError):
            fn(y_np)
        with pytest.raises(ValueError):
            fn(y_pd)


def test_missing_values_in_y_are_rejected():
    """Ensure missing labels are rejected."""
    y_np = np.array([0, 1, np.nan], dtype=float)
    y_pd = pd.Series([0, 1, pd.NA], dtype="Float64")

    for fn in (check_classification_y, check_regression_y, check_anomaly_detection_y):
        with pytest.raises(ValueError):
            fn(y_np)
        with pytest.raises(ValueError):
            fn(y_pd)
