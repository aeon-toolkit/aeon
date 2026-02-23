"""Test cases for the range-based anomaly detection metrics."""

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection._range_metrics import (
    range_f_score,
    range_precision,
    range_recall,
)


def _execute_test_case(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected_precision: float,
    expected_recall: float,
    expected_f1: float,
    name: str = "test case",
    cardinality: str = "one",
    bias: str = "flat",
    floating_precision: int = 6,
) -> None:
    precision = range_precision(y_true, y_pred, cardinality=cardinality, bias=bias)
    recall = range_recall(y_true, y_pred, cardinality=cardinality, bias=bias)
    f1_score = range_f_score(
        y_true,
        y_pred,
        cardinality=cardinality,
        p_bias=bias,
        r_bias=bias,
        p_alpha=0.0,
        r_alpha=0.0,
        beta=1.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=floating_precision,
        err_msg=(
            f"Precision failed for {name}! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=floating_precision,
        err_msg=(
            f"Recall failed for {name}! " f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=floating_precision,
        err_msg=(
            f"F1-Score failed for {name}! " f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_single_overlapping_range():
    """Test for single overlapping range."""
    _execute_test_case(
        y_true=np.array([0, 0, 1, 1, 1, 1, 1]),
        y_pred=np.array([0, 1, 1, 1, 1, 0, 0]),
        expected_precision=0.750000,
        expected_recall=0.600000,
        expected_f1=0.666667,
        name="single overlapping range",
    )


def test_multiple_non_overlapping_ranges():
    """Test for multiple non-overlapping ranges."""
    _execute_test_case(
        y_true=np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]),
        expected_precision=0.000000,
        expected_recall=0.000000,
        expected_f1=0.000000,
        name="multiple non-overlapping range",
    )


def test_multiple_overlapping_ranges():
    """Test for multiple overlapping ranges."""
    _execute_test_case(
        y_true=np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]),
        y_pred=np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]),
        expected_precision=0.666667,
        expected_recall=0.400000,
        expected_f1=0.500000,
        name="multiple overlapping ranges",
    )


def test_nested_lists_of_predictions():
    """Test for nested lists of predictions."""
    _execute_test_case(
        y_true=np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]),
        y_pred=np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1]),
        expected_precision=0.555556,
        expected_recall=0.566667,
        expected_f1=0.561056,
        name="nested lists of predictions",
    )


def test_all_encompassing_range():
    """Test for all encompassing range."""
    _execute_test_case(
        y_true=np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]),
        y_pred=np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        expected_precision=0.600000,
        expected_recall=1.000000,
        expected_f1=0.750000,
        name="all encompassing range",
    )


def test_multiple_overlapping_ranges_with_gamma_reciprocal():
    """Test for multiple overlapping ranges with gamma=reciprocal."""
    _execute_test_case(
        y_true=np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]),
        y_pred=np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]),
        expected_precision=0.666667,
        expected_recall=0.200000,
        expected_f1=0.307692,
        name="multiple overlapping ranges with reciprocal cardinality",
        cardinality="reciprocal",
    )


def test_multiple_overlapping_ranges_with_bias_middle():
    """Test for multiple overlapping ranges with bias_type=middle."""
    _execute_test_case(
        y_true=np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]),
        y_pred=np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]),
        expected_precision=0.750000,
        expected_recall=0.333333,
        expected_f1=0.461538,
        name="multiple overlapping ranges with middle bias",
        bias="middle",
    )


def test_multiple_overlapping_ranges_with_bias_middle_gamma_reciprocal():
    """Test for multiple overlapping ranges with bias_type=middle, gamma=reciprocal."""
    _execute_test_case(
        y_true=np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]),
        y_pred=np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]),
        expected_precision=0.750000,
        expected_recall=0.166667,
        expected_f1=0.272727,
        name="multiple overlapping ranges with middle bias and reciprocal cardinality",
        cardinality="reciprocal",
        bias="middle",
    )
