"""Test cases for the range-based anomaly detection metrics."""

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection._range_metrics import (
    range_f_score,
    range_precision,
    range_recall,
)
from aeon.benchmarking.metrics.anomaly_detection._range_ts_metrics import (
    ts_fscore,
    ts_precision,
    ts_recall,
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


# TODO: remove in v1.3.0
def test_range_based_input():
    """Test with input being range-based or binary-based."""
    y_pred_range = [(1, 2)]
    y_true_range = [(1, 1)]
    y_pred_binary = np.array([0, 1, 1, 0])
    y_true_binary = np.array([0, 1, 0, 0])

    expected_precision = 0.5
    expected_recall = 1.000000
    expected_f1 = 0.666667

    # for range-based input
    precision_range = ts_precision(
        y_pred_range, y_true_range, gamma="reciprocal", bias_type="flat"
    )
    recall_range = ts_recall(
        y_pred_range,
        y_true_range,
        gamma="reciprocal",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score_range = ts_fscore(
        y_pred_range,
        y_true_range,
        gamma="reciprocal",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision_range,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision mismatch: "
            f"ts_precision={precision_range} vs"
            f"expected_precision_range={expected_precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall_range,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall mismatch: "
            f"ts_recall={recall_range} vs expected_recall_range={expected_recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score_range,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score mismatch: "
            f"ts_fscore={f1_score_range} vs expected_f_score_range={expected_f1}"
        ),
    )

    # for binary input
    _execute_test_case(
        y_true=y_true_binary,
        y_pred=y_pred_binary,
        expected_precision=expected_precision,
        expected_recall=expected_recall,
        expected_f1=expected_f1,
        name="binary input is inconsistent with range input",
        cardinality="reciprocal",
        bias="flat",
    )
