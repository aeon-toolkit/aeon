"""Test cases for the range-based anomaly detection metrics."""

import numpy as np
import pytest

from aeon.benchmarking.metrics.anomaly_detection import (
    range_f_score,
    range_precision,
    range_recall,
)
from aeon.benchmarking.metrics.anomaly_detection.range_metrics import (
    _binary_to_ranges,
    ts_fscore,
    ts_precision,
    ts_recall,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_single_overlapping_range():
    """Test for single overlapping range."""
    y_pred_bin = np.array([0, 1, 1, 1, 1, 0, 0])
    y_real_bin = np.array([0, 0, 1, 1, 1, 1, 1])

    precision = ts_precision(y_pred_bin, y_real_bin, gamma="one", bias_type="flat")
    recall = ts_recall(y_pred_bin, y_real_bin, gamma="one", bias_type="flat", alpha=0.0)
    f1_score = ts_fscore(
        y_pred_bin,
        y_real_bin,
        gamma="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    rb_prec = range_precision(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_rec = range_recall(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_multiple_non_overlapping_ranges():
    """Test for multiple non-overlapping ranges."""
    y_pred_bin = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    y_real_bin = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])

    precision = ts_precision(y_pred_bin, y_real_bin, gamma="one", bias_type="flat")
    recall = ts_recall(y_pred_bin, y_real_bin, gamma="one", bias_type="flat", alpha=0.0)
    f1_score = ts_fscore(
        y_pred_bin,
        y_real_bin,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    rb_prec = range_precision(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_rec = range_recall(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_multiple_overlapping_ranges():
    """Test for multiple overlapping ranges."""
    y_pred_bin = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real_bin = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])

    precision = ts_precision(y_pred_bin, y_real_bin, gamma="one", bias_type="flat")
    recall = ts_recall(y_pred_bin, y_real_bin, gamma="one", bias_type="flat", alpha=0.0)
    f1_score = ts_fscore(
        y_pred_bin,
        y_real_bin,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    rb_prec = range_precision(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_rec = range_recall(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_nested_lists_of_predictions():
    """Test for nested lists of predictions."""
    y_pred_bin = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    y_real_bin = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0])

    precision = ts_precision(y_pred_bin, y_real_bin, gamma="one", bias_type="flat")
    recall = ts_recall(y_pred_bin, y_real_bin, gamma="one", bias_type="flat", alpha=0.0)
    f1_score = ts_fscore(
        y_pred_bin,
        y_real_bin,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    rb_prec = range_precision(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_rec = range_recall(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_all_encompassing_range():
    """Test for all encompassing range."""
    y_pred_bin = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_real_bin = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

    precision = ts_precision(y_pred_bin, y_real_bin, gamma="one", bias_type="flat")
    recall = ts_recall(y_pred_bin, y_real_bin, gamma="one", bias_type="flat", alpha=0.0)
    f1_score = ts_fscore(
        y_pred_bin,
        y_real_bin,
        gamma="one",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    rb_prec = range_precision(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_rec = range_recall(y_real_bin, y_pred_bin, cardinality="one", bias="flat")
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="one",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_binary_input_example():
    """Comparing prts package(_binary.py) and range-based metrics(range_metrics.py)."""
    y_pred_binary = np.array([0, 1, 1, 0])
    y_true_binary = np.array([0, 1, 0, 0])

    precision = ts_precision(
        y_pred_binary, y_true_binary, gamma="reciprocal", bias_type="flat"
    )
    recall = ts_recall(
        y_pred_binary, y_true_binary, gamma="reciprocal", bias_type="flat", alpha=0.0
    )
    f1_score = ts_fscore(
        y_pred_binary,
        y_true_binary,
        beta=1,
        gamma="reciprocal",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    rb_prec = range_precision(
        y_true_binary, y_pred_binary, cardinality="reciprocal", bias="flat"
    )
    rb_rec = range_recall(
        y_true_binary, y_pred_binary, cardinality="reciprocal", bias="flat"
    )
    rb_fsc = range_f_score(
        y_true_binary,
        y_pred_binary,
        beta=1,
        cardinality="reciprocal",
        p_bias="flat",
        r_bias="flat",
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_multiple_overlapping_ranges_with_gamma_reciprocal():
    """Test for multiple overlapping ranges with gamma=reciprocal."""
    y_pred_bin = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real_bin = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])

    precision = ts_precision(
        y_pred_bin, y_real_bin, gamma="reciprocal", bias_type="flat"
    )
    recall = ts_recall(
        y_pred_bin, y_real_bin, gamma="reciprocal", bias_type="flat", alpha=0.0
    )
    f1_score = ts_fscore(
        y_pred_bin,
        y_real_bin,
        gamma="reciprocal",
        beta=1,
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    rb_prec = range_precision(
        y_real_bin, y_pred_bin, cardinality="reciprocal", bias="flat"
    )
    rb_rec = range_recall(y_real_bin, y_pred_bin, cardinality="reciprocal", bias="flat")
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="reciprocal",
        p_bias="flat",
        r_bias="flat",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_multiple_overlapping_ranges_with_bias_middle():
    """Test for multiple overlapping ranges with bias_type=middle using range-binary conversion."""  # noqa E501
    y_pred_bin = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real_bin = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])

    # metric functions can also handle range-based input (internal binary conversion)
    y_pred = _binary_to_ranges(
        y_pred_bin
    )  # Convert binary to range to show compatibility
    y_real = _binary_to_ranges(y_real_bin)

    precision = ts_precision(y_pred, y_real, gamma="one", bias_type="middle")
    recall = ts_recall(y_pred, y_real, gamma="one", bias_type="middle", alpha=0.0)
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

    rb_prec = range_precision(y_real_bin, y_pred_bin, cardinality="one", bias="middle")
    rb_rec = range_recall(y_real_bin, y_pred_bin, cardinality="one", bias="middle")
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="one",
        p_bias="middle",
        r_bias="middle",
        p_alpha=0.0,
        r_alpha=0.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_multiple_overlapping_ranges_with_bias_middle_gamma_reciprocal():
    """Test for multiple overlapping ranges with bias_type=middle, gamma=reciprocal."""
    y_pred_bin = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    y_real_bin = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])

    precision = ts_precision(
        y_pred_bin, y_real_bin, gamma="reciprocal", bias_type="middle"
    )
    recall = ts_recall(
        y_pred_bin, y_real_bin, gamma="reciprocal", bias_type="middle", alpha=1.0
    )
    f1_score = ts_fscore(
        y_pred_bin,
        y_real_bin,
        gamma="reciprocal",
        beta=1,
        p_bias="middle",
        r_bias="middle",
        p_alpha=0.0,
        r_alpha=1.0,
    )

    rb_prec = range_precision(
        y_real_bin, y_pred_bin, cardinality="reciprocal", bias="middle"
    )
    rb_rec = range_recall(
        y_real_bin, y_pred_bin, cardinality="reciprocal", bias="middle", alpha=1.0
    )
    rb_fsc = range_f_score(
        y_real_bin,
        y_pred_bin,
        beta=1,
        cardinality="reciprocal",
        p_bias="middle",
        r_bias="middle",
        p_alpha=0.0,
        r_alpha=1.0,
    )

    np.testing.assert_almost_equal(
        precision,
        rb_prec,
        decimal=6,
        err_msg=f"Precision mismatch: ts_precision={precision} vs prts={rb_prec}",
    )
    np.testing.assert_almost_equal(
        recall,
        rb_rec,
        decimal=6,
        err_msg=f"Recall mismatch: ts_recall={recall} vs prts={rb_rec}",
    )
    np.testing.assert_almost_equal(
        f1_score,
        rb_fsc,
        decimal=6,
        err_msg=f"F1-Score mismatch: ts_fscore={f1_score} vs prts={rb_fsc}",
    )
