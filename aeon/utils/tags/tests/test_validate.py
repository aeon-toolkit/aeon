"""Tests for tag validation functions."""

import pytest

from aeon.classification import BaseClassifier
from aeon.testing.mock_estimators import MockClassifier
from aeon.testing.mock_estimators._mock_anomaly_detectors import MockAnomalyDetector
from aeon.utils.tags import check_valid_tags


@pytest.mark.parametrize(
    "estimator",
    [
        MockClassifier,
        MockClassifier(),
        BaseClassifier,
        MockAnomalyDetector,
        MockAnomalyDetector(),
    ],
)
def test_check_valid_tags(estimator):
    """Test the check_valid_tags function."""
    check_valid_tags(estimator)

    tags = {
        "python_version": ">3.8",
        "python_dependencies": None,
        "non-deterministic": False,
    }

    check_valid_tags(estimator, tags=tags, error_on_missing=False)


def test_check_valid_tags_invalid():
    """Test the check_valid_tags function with invalid tags and input."""
    pass
