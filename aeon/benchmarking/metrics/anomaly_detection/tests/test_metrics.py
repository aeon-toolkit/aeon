"""Test cases for metrics."""

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection.range_metrics import (
    ts_fscore,
    ts_precision,
    ts_recall,
)

# Single Overlapping Range
y_pred = [(1, 4)]
y_real = [(2, 6)]

precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
recall = ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)
f1_score = ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)

np.testing.assert_almost_equal(precision, 0.750000, decimal=6)
np.testing.assert_almost_equal(recall, 0.600000, decimal=6)
np.testing.assert_almost_equal(f1_score, 0.666667, decimal=6)

# Multiple Non-Overlapping Ranges
y_pred = [(1, 2), (7, 8)]
y_real = [(3, 4), (9, 10)]

precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
recall = ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)
f1_score = ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)

np.testing.assert_almost_equal(precision, 0.000000, decimal=6)
np.testing.assert_almost_equal(recall, 0.000000, decimal=6)
np.testing.assert_almost_equal(f1_score, 0.000000, decimal=6)

# Multiple Overlapping Ranges
y_pred = [(1, 3), (5, 7)]
y_real = [(2, 6), (8, 10)]

precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
recall = ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)
f1_score = ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)

np.testing.assert_almost_equal(precision, 0.666667, decimal=6)
np.testing.assert_almost_equal(recall, 0.5, decimal=6)
np.testing.assert_almost_equal(f1_score, 0.571429, decimal=6)

# Nested Lists of Predictions
y_pred = [[(1, 3), (5, 7)], [(10, 12)]]
y_real = [(2, 6), (8, 10)]

precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
recall = ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)
f1_score = ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)

np.testing.assert_almost_equal(precision, 0.555556, decimal=6)
np.testing.assert_almost_equal(recall, 0.555556, decimal=6)
np.testing.assert_almost_equal(f1_score, 0.555556, decimal=6)

# All Encompassing Range
y_pred = [(1, 10)]
y_real = [(2, 3), (5, 6), (8, 9)]

precision = ts_precision(y_pred, y_real, gamma="one", bias_type="flat")
recall = ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)
f1_score = ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0)

np.testing.assert_almost_equal(precision, 0.600000, decimal=6)
np.testing.assert_almost_equal(recall, 1.000000, decimal=6)
np.testing.assert_almost_equal(f1_score, 0.75, decimal=6)
