"""Whole-series anomaly detection methods."""

__all__ = [
    "BaseCollectionAnomalyDetector",
    "ClassificationAdapter",
    "OutlierDetectionAdapter",
    "ROCKAD",
]

from aeon.anomaly_detection.whole_series._classification import ClassificationAdapter
from aeon.anomaly_detection.whole_series._outlier_detection import (
    OutlierDetectionAdapter,
)
from aeon.anomaly_detection.whole_series._rockad import ROCKAD
from aeon.anomaly_detection.whole_series.base import BaseCollectionAnomalyDetector
