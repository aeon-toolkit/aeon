"""Tests for the series anomaly detection estimator checks."""

__maintainer__ = []

import numpy as np
import pytest

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.testing.estimator_checking._yield_series_anomaly_detection_checks import (
    check_series_anomaly_detector_discriminates,
    check_series_anomaly_detector_output,
)

DATATYPE = "UnivariateSeries-Anomaly-np.ndarray"


class _ConstantScoreDetector(BaseSeriesAnomalyDetector):
    """Detector which returns the same anomaly score at every time point."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "anomaly_output_type": "anomaly_scores",
        "learning_type:unsupervised": True,
    }

    def __init__(self):
        super().__init__(axis=1)

    def _predict(self, X):
        return np.full(X.shape[self.axis], 0.7)


def test_constant_scores_fail_discrimination_check():
    """Test a constant anomaly score is rejected on the labelled fixture.

    A constant score is only wrong because the fixture used here holds labelled
    anomalies. On a constant input series a constant score would be correct, and this
    check never runs on such a series.
    """
    detector = _ConstantScoreDetector()

    # the existing output check only looks at shape and dtype, so it accepts this
    check_series_anomaly_detector_output(detector, DATATYPE)

    with pytest.raises(AssertionError, match="do not separate the labelled anomalies"):
        check_series_anomaly_detector_discriminates(detector, DATATYPE)
