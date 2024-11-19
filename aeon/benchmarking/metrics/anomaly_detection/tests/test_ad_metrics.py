"""Tests for metrics in anomaly_detection module."""

import numpy as np
import pytest

from aeon.benchmarking.metrics.anomaly_detection import (
    f_score_at_k_points,
    f_score_at_k_ranges,
    pr_auc_score,
    range_f_score,
    range_pr_auc_score,
    range_pr_vus_score,
    range_precision,
    range_recall,
    range_roc_auc_score,
    range_roc_vus_score,
    roc_auc_score,
    rp_rr_auc_score,
)
from aeon.testing.data_generation import make_example_1d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

pr_metrics = [pr_auc_score]
range_metrics = [
    range_roc_auc_score,
    range_pr_auc_score,
    range_roc_vus_score,
    range_pr_vus_score,
]
other_metrics = [
    roc_auc_score,
    f_score_at_k_points,
]
continuous_metrics = [*pr_metrics, *other_metrics, *range_metrics]
binary_metrics = []

if _check_soft_dependencies("prts", severity="none"):
    pr_metrics.append(rp_rr_auc_score)
    range_metrics.extend(
        [
            range_recall,
            range_precision,
            range_f_score,
        ]
    )
    other_metrics.extend(
        [
            f_score_at_k_ranges,
            rp_rr_auc_score,
        ]
    )
    continuous_metrics.extend(
        [
            rp_rr_auc_score,
            f_score_at_k_ranges,
        ]
    )
    binary_metrics = [range_recall, range_precision, range_f_score]

metrics = [*pr_metrics, *range_metrics, *other_metrics]


@pytest.mark.parametrize("metric", metrics, ids=[m.__name__ for m in metrics])
def test_metric_output(metric):
    """Test output has correct format."""
    y_score = make_example_1d_numpy(n_timepoints=20, random_state=0)
    y_true = make_example_1d_numpy(n_timepoints=20, random_state=1)
    y_true = (y_true > 0.5).astype(int)

    if metric in continuous_metrics:
        res = metric(
            y_true=y_true,
            y_score=y_score,
        )
    else:
        y_pred = (y_score > 0.5).astype(int)
        res = metric(
            y_true=y_true,
            y_pred=y_pred,
        )

    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0


@pytest.mark.parametrize(
    "metric", continuous_metrics, ids=[m.__name__ for m in continuous_metrics]
)
def test_continuous_metric_requires_scores(metric):
    """Test that continuous metrics require float scores."""
    y_scores = np.array([1, 0, 1, 0, 0])
    y_true = np.array([0, 0, 1, 0, 0])

    with pytest.raises(ValueError) as ex:
        metric(y_true, y_scores)
    assert "scores must be floats" in str(ex.value)

    y_scores = y_scores.astype(np.float64)
    metric(y_true, y_scores)
    assert True, "Metric should accept float, so no error should be raised"


@pytest.mark.parametrize(
    "metric", binary_metrics, ids=[m.__name__ for m in binary_metrics]
)
def test_binary_metric_requires_predictions(metric):
    """Test that binary metrics require integer or boolean predictions."""
    y_scores = np.array([0.8, 0.1, 0.9, 0.3, 0.3], dtype=np.float64)
    y_true = np.array([0, 0, 1, 0, 0], dtype=np.bool_)

    with pytest.raises(ValueError) as ex:
        metric(y_true, y_scores)
    assert "scores must be integers" in str(ex.value)

    y_pred = np.array(y_scores > 0.5, dtype=np.int_)
    metric(y_true, y_pred)

    y_pred = y_pred.astype(np.bool_)
    metric(y_true, y_pred)
    assert True, "Metric should accept ints and bools, so no error should be raised"


@pytest.mark.parametrize("metric", metrics, ids=[m.__name__ for m in metrics])
def test_edge_cases(metric):
    """Test edge cases for all metrics."""
    y_true = np.zeros(10, dtype=np.int_)
    y_true[2:4] = 1
    y_true[6:8] = 1
    y_zeros = np.zeros_like(y_true, dtype=np.float64)
    y_flat = np.full_like(y_true, fill_value=0.5, dtype=np.float64)
    y_ones = np.ones_like(y_true, dtype=np.float64)

    for y_pred in [y_zeros, y_flat, y_ones]:
        if metric in binary_metrics:
            score = metric(y_true, y_pred.astype(np.int_))
        else:
            score = metric(y_true, y_pred)
        np.testing.assert_almost_equal(score, 0)


@pytest.mark.parametrize("metric", pr_metrics, ids=[m.__name__ for m in pr_metrics])
def test_edge_cases_pr_metrics(metric):
    """Test edge cases for PR metrics."""
    y_true = np.zeros(10, dtype=np.int_)
    y_true[2:4] = 1
    y_true[6:8] = 1
    y_inverted = (y_true * -1 + 1).astype(np.float64)

    score = metric(y_true, y_inverted)
    assert score <= 0.2, f"{metric.__name__}(y_true, y_inverted)={score} is not <= 0.2"


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_range_based_f1():
    """Test range-based F1 score."""
    y_pred = np.array([0, 1, 1, 0])
    y_true = np.array([0, 1, 0, 0])
    result = range_f_score(y_true, y_pred)
    np.testing.assert_almost_equal(result, 0.66666, decimal=4)


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_range_based_precision():
    """Test range-based precision."""
    y_pred = np.array([0, 1, 1, 0])
    y_true = np.array([0, 1, 0, 0])
    result = range_precision(y_true, y_pred)
    assert result == 0.5


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_range_based_recall():
    """Test range-based recall."""
    y_pred = np.array([0, 1, 1, 0])
    y_true = np.array([0, 1, 0, 0])
    result = range_recall(y_true, y_pred)
    assert result == 1


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_rf1_value_error():
    """Test range-based F1 score raises ValueError on binary predictions."""
    y_pred = np.array([0, 0.2, 0.7, 0])
    y_true = np.array([0, 1, 0, 0])
    with pytest.raises(ValueError):
        range_f_score(y_true, y_pred)


def test_pr_curve_auc():
    """Test PR curve AUC."""
    y_pred = np.array([0, 0.1, 1.0, 0.5, 0, 0])
    y_true = np.array([0, 0, 1, 1, 0, 0])
    result = pr_auc_score(y_true, y_pred)
    np.testing.assert_almost_equal(result, 1.0000, decimal=4)


# def test_average_precision():
#     y_pred = np.array([0, 0.1, 1., .5, 0, 0])
#     y_true = np.array([0, 1, 1, 0, 0, 0])
#     result = average_precision_score(y_true, y_pred)
#     np.testing.assert_almost_equal(result, 0.8333, decimal=4)


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_range_based_p_range_based_r_curve_auc():
    """Test range-based precision-recall curve AUC."""
    y_pred = np.array([0, 0.1, 1.0, 0.5, 0.1, 0])
    y_true = np.array([0, 1, 1, 1, 0, 0])
    result = rp_rr_auc_score(y_true, y_pred)
    np.testing.assert_almost_equal(result, 0.9792, decimal=4)


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_range_based_p_range_based_r_auc_perfect_hit():
    """Test range-based precision-recall curve AUC with perfect hit."""
    y_pred = np.array([0, 0, 0.5, 0.5, 0, 0])
    y_true = np.array([0, 0, 1, 1, 0, 0])
    result = rp_rr_auc_score(y_true, y_pred)
    np.testing.assert_almost_equal(result, 1.0000, decimal=4)


@pytest.mark.skipif(
    not _check_soft_dependencies("prts", severity="none"),
    reason="required soft dependency prts not available",
)
def test_f_score_at_k_ranges():
    """Test range-based F1 score at k ranges."""
    y_pred = np.array([0.4, 0.1, 1.0, 0.5, 0.1, 0, 0.4, 0.5])
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    result = f_score_at_k_ranges(y_true, y_pred)
    np.testing.assert_almost_equal(result, 0.5000, decimal=4)
    result = f_score_at_k_ranges(y_true, y_pred, k=3)
    np.testing.assert_almost_equal(result, 0.8000, decimal=4)


def test_fscore_at_k_points():
    """Test F1 score at k points."""
    y_pred = np.array([0.4, 0.6, 1.0, 0.5, 0.1, 0, 0.5, 0.4])
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    result = f_score_at_k_points(y_true, y_pred)
    np.testing.assert_almost_equal(result, 1.0000, decimal=4)
    result = f_score_at_k_points(y_true, y_pred, k=1)
    np.testing.assert_almost_equal(result, 0.4000, decimal=4)


# VUS fixtures
y_true = np.zeros(200)
y_true[10:20] = 1
y_true[28:33] = 1
y_true[110:120] = 1
y_score = np.random.default_rng(41).random(200) * 0.5
y_score[16:22] = 1
y_score[33:38] = 1
y_score[160:170] = 1


def test_range_pr_auc_compat():
    """Test range-based PR AUC."""
    result = range_pr_auc_score(y_true, y_score)
    np.testing.assert_almost_equal(result, 0.3737854660, decimal=10)


def test_range_roc_auc_compat():
    """Test range-based ROC AUC."""
    result = range_roc_auc_score(y_true, y_score)
    np.testing.assert_almost_equal(result, 0.7108527197, decimal=10)


def test_edge_case_existence_reward_compat():
    """Test edge case for existence reward in range-based PR/ROC AUC."""
    result = range_pr_auc_score(y_true, y_score, buffer_size=4)
    np.testing.assert_almost_equal(result, 0.2506464391, decimal=10)
    result = range_roc_auc_score(y_true, y_score, buffer_size=4)
    np.testing.assert_almost_equal(result, 0.6143220816, decimal=10)


def test_range_pr_volume_compat():
    """Test range-based PR volume under the curve."""
    result = range_pr_vus_score(y_true, y_score, max_buffer_size=200)
    np.testing.assert_almost_equal(result, 0.7493254559, decimal=10)


def test_range_roc_volume_compat():
    """Test range-based ROC volume under the curve."""
    result = range_roc_vus_score(y_true, y_score, max_buffer_size=200)
    np.testing.assert_almost_equal(result, 0.8763382130, decimal=10)
