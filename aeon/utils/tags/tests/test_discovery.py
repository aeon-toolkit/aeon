"""Tests for tag lookup functions."""

import pytest

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.classification import BaseClassifier
from aeon.testing.mock_estimators import MockClassifier
from aeon.testing.mock_estimators._mock_anomaly_detectors import MockAnomalyDetector
from aeon.utils.tags import ESTIMATOR_TAGS, all_tags_for_estimator


def test_all_tags_for_estimator_classification():
    """Test the all_tags_for_estimator function with classification estimators."""
    clf = MockClassifier()
    tags = all_tags_for_estimator(clf)

    assert isinstance(tags, dict)
    assert "python_version" in tags
    assert "non_deterministic" in tags
    assert "capability:unequal_length" in tags
    assert "capability:contractable" in tags

    assert tags == all_tags_for_estimator(MockClassifier)
    assert tags == all_tags_for_estimator(BaseClassifier)
    assert tags == all_tags_for_estimator("classifier")

    tag_names = all_tags_for_estimator(MockClassifier(), names_only=True)
    assert isinstance(tag_names, list)
    assert tag_names == list(tags.keys())
    assert set(tag_names).issubset(ESTIMATOR_TAGS.keys())


def test_all_tags_for_estimator_anomaly_detection():
    """Test the all_tags_for_estimator function with anomaly detection estimators."""
    ad = MockAnomalyDetector()
    tags = all_tags_for_estimator(ad)

    assert isinstance(tags, dict)
    assert "python_version" in tags
    assert "non_deterministic" in tags
    assert "capability:unequal_length" not in tags
    assert "capability:contractable" not in tags

    assert tags == all_tags_for_estimator(MockAnomalyDetector)
    assert tags == all_tags_for_estimator(BaseAnomalyDetector)
    assert tags == all_tags_for_estimator("anomaly-detector")

    tag_names = all_tags_for_estimator(MockAnomalyDetector(), names_only=True)
    assert isinstance(tag_names, list)
    assert tag_names == list(tags.keys())
    assert set(tag_names).issubset(ESTIMATOR_TAGS.keys())


def test_all_tags_for_estimator_invalid():
    """Test the all_tags_for_estimator function with invalid input."""
    with pytest.raises(ValueError, match="Estimator must be an"):
        all_tags_for_estimator("invalid")
