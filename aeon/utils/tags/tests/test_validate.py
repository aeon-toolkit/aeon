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
        "non_deterministic": False,
    }

    check_valid_tags(estimator, tags=tags, error_on_missing=False)


def test_check_valid_tags_invalid():
    """Test the check_valid_tags function with invalid tags and input."""
    # invalid estimator
    with pytest.raises(ValueError, match="Estimator must be an"):
        check_valid_tags("estimator")

    # invalid tag
    tags = {
        "python_version": ">3.8",
        "python_dependencies": None,
        "fake-tag": 6,
    }

    with pytest.raises(ValueError, match="Tag fake-tag is not a valid tag"):
        check_valid_tags(MockClassifier, tags=tags)

    # invalid tag type for classifier
    tags = {
        "python_version": ">3.8",
        "python_dependencies": None,
        "returns_dense": False,
    }

    with pytest.raises(ValueError, match="Tag returns_dense is not compatible"):
        check_valid_tags(MockClassifier, tags=tags)

    # invalid tag value in tag
    tags = {
        "python_version": ">3.8",
        "python_dependencies": None,
        "non_deterministic": 1,
    }

    with pytest.raises(
        ValueError, match="Value 1 is not a valid value for tag non_deterministic"
    ):
        check_valid_tags(MockClassifier, tags=tags)
