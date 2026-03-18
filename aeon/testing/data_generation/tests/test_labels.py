"""Tests for label generation functions."""

import numpy as np

from aeon.testing.data_generation import make_anomaly_detection_labels


def test_make_anomaly_detection_labels():
    """Test anomaly detection label generation."""
    labels = make_anomaly_detection_labels(50)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (50,)
    assert np.all(np.isin(labels, [0, 1]))
    assert np.sum(labels) == 1

    labels2 = make_anomaly_detection_labels(100, anomaly_rate=0.2)

    assert isinstance(labels2, np.ndarray)
    assert labels2.shape == (100,)
    assert np.all(np.isin(labels2, [0, 1]))
    assert np.sum(labels2) == 20
