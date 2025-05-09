"""Whole-series anomaly detection methods."""

__all__ = [
    "BaseCollectionAnomalyDetector",
    "ClassificationAdapter",
    "OutlierDetectionAdapter",
]

from aeon.anomaly_detection.collection._classification import ClassificationAdapter
from aeon.anomaly_detection.collection._outlier_detection import OutlierDetectionAdapter
from aeon.anomaly_detection.collection.base import BaseCollectionAnomalyDetector
