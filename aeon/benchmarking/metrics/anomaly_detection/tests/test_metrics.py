"""Test cases for the range-based anomaly detection metrics."""

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection.range_metrics import (
    ts_fscore,
    ts_precision,
    ts_recall,
)


def test_single_overlapping_range():
    """Test for single overlapping range."""
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0])
    y_real = np.array([0, 0, 1, 1, 1, 1, 1])
    expected_precision = 0.750000
    expected_recall = 0.600000
    expected_f1 = 0.666667

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="one",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for single overlapping range! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for single overlapping range! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for single overlapping range! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_non_overlapping_ranges():
    """Test for multiple non-overlapping ranges."""
    y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    y_real = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])

    expected_precision = 0.000000
    expected_recall = 0.000000
    expected_f1 = 0.000000

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="one",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple non-overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple non-overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple non-overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_overlapping_ranges():
    """Test for multiple overlapping ranges."""
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])

    expected_precision = 0.666667
    expected_recall = 0.400000
    expected_f1 = 0.500000

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="one",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_nested_lists_of_predictions():
    """Test for nested lists of predictions."""
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    y_real = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0])

    expected_precision = 0.555556
    expected_recall = 0.566667
    expected_f1 = 0.561056

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="one",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for nested lists of predictions! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for nested lists of predictions! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for nested lists of predictions! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_all_encompassing_range():
    """Test for all encompassing range."""
    y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_real = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

    expected_precision = 0.600000
    expected_recall = 1.000000
    expected_f1 = 0.750000

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="one",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for all encompassing range! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for all encompassing range! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for all encompassing range! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_range_based_input():
    """Test with input being range-based or bianry-based."""
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
    precision_binary = ts_precision(
        y_pred_binary, y_true_binary, gamma="reciprocal", bias_type="flat"
    )
    recall_binary = ts_recall(
        y_pred_binary,
        y_true_binary,
        gamma="reciprocal",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score_binary = ts_fscore(
        y_pred_binary,
        y_true_binary,
        gamma="reciprocal",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision_binary,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision mismatch: "
            f"ts_precision={precision_range} vs "
            f"expected_precision_binary={expected_precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall_binary,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall mismatch: "
            f"ts_recall={recall_range} vs expected_recall_binary={expected_recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score_binary,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score mismatch: "
            f"ts_fscore={f1_score_range} vs expected_f_score_binary={expected_f1}"
        ),
    )


def test_multiple_overlapping_ranges_with_gamma_reciprocal():
    """Test for multiple overlapping ranges with gamma=reciprocal."""
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    expected_precision = 0.666667
    expected_recall = 0.200000
    expected_f1 = 0.307692

    precision = ts_precision(y_pred, y_real, gamma="reciprocal", bias_type="flat")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="reciprocal",
        bias_type="flat",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="reciprocal",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_overlapping_ranges_with_bias_middle():
    """Test for multiple overlapping ranges with bias_type=middle."""
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    expected_precision = 0.750000
    expected_recall = 0.333333
    expected_f1 = 0.461538

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="middle")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="one",
        bias_type="middle",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="one",
        beta=1,
        p_bias="middle",
        r_bias="middle",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )


def test_multiple_overlapping_ranges_with_bias_middle_gamma_reciprocal():
    """Test for multiple overlapping ranges with bias_type=middle, gamma=reciprocal."""
    y_pred = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    expected_precision = 0.750000
    expected_recall = 0.166667
    expected_f1 = 0.272727

    precision = ts_precision(y_pred, y_real, gamma="reciprocal", bias_type="middle")
    recall = ts_recall(
        y_pred,
        y_real,
        gamma="reciprocal",
        bias_type="middle",
        alpha=0.0,
    )
    f1_score = ts_fscore(
        y_pred,
        y_real,
        gamma="reciprocal",
        beta=1,
        p_bias="middle",
        r_bias="middle",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        expected_precision,
        decimal=6,
        err_msg=(
            f"Precision failed for multiple overlapping ranges! "
            f"Expected={expected_precision}, Got={precision}"
        ),
    )
    np.testing.assert_almost_equal(
        recall,
        expected_recall,
        decimal=6,
        err_msg=(
            f"Recall failed for multiple overlapping ranges! "
            f"Expected={expected_recall}, Got={recall}"
        ),
    )
    np.testing.assert_almost_equal(
        f1_score,
        expected_f1,
        decimal=6,
        err_msg=(
            f"F1-Score failed for multiple overlapping ranges! "
            f"Expected={expected_f1}, Got={f1_score}"
        ),
    )
