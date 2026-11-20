"""Regression tests for segmentation estimator checks."""

import pytest

from aeon.testing.estimator_checking._yield_segmentation_checks import (
    check_segmenter_base_functionality,
)
from aeon.testing.mock_estimators import MockSegmenter


def test_segmenter_fit_is_empty_contract(monkeypatch):
    """Reject a segmenter that declares an implemented fit method empty."""
    check_segmenter_base_functionality(MockSegmenter)
    monkeypatch.setitem(MockSegmenter._tags, "fit_is_empty", True)
    with pytest.raises(AssertionError, match="fit_is_empty"):
        check_segmenter_base_functionality(MockSegmenter)
