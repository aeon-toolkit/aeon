"""Test anomaly detection thresholding methods."""

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection.thresholding import (
    percentile_threshold,
    sigma_threshold,
    top_k_points_threshold,
    top_k_ranges_threshold,
)

y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
y_score = np.array([np.nan, 0.1, 0.9, 0.5, 0.6, 0.55, 0.2, 0.1, np.nan])


def test_percentile_thresholding():
    """Test percentile thresholding."""
    threshold = percentile_threshold(y_score, percentile=90)
    np.testing.assert_almost_equal(threshold, 0.72, decimal=2)


def test_top_k_points_thresholding():
    """Test top k points thresholding."""
    threshold = top_k_points_threshold(y_true, y_score, k=2)
    np.testing.assert_almost_equal(threshold, 0.58, decimal=2)


def test_top_k_ranges_thresholding():
    """Test top k ranges thresholding."""
    threshold = top_k_ranges_threshold(y_true, y_score, k=2)
    np.testing.assert_almost_equal(threshold, 0.60, decimal=2)


def test_sigma_thresholding():
    """Test sigma thresholding."""
    threshold = sigma_threshold(y_score, factor=1)
    np.testing.assert_almost_equal(threshold, 0.70, decimal=2)
