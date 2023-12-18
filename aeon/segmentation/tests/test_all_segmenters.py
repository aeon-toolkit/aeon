"""Tests for all segmenters."""
import pytest

from aeon.registry import all_estimators

ALL_SEGMENTERS = all_estimators(
    estimator_types="segmenter",
    return_names=False,
)


@pytest.mark.parametrize("segmenter", ALL_SEGMENTERS)
def test_segmenter_interface(segmenter):
    """Test does not override final methods."""
    assert "fit" not in segmenter.__dict__
    assert "predict" not in segmenter.__dict__
    # Test that all segmenters implement abstract predict.
    assert "_predict" in segmenter.__dict__
    # Test that fit_is_empty is correctly set
    fit_is_empty = segmenter.get_class_tag(tag_name="fit_is_empty")
    assert not fit_is_empty == "_fit" not in segmenter.__dict__
    # Test that capability:multivariate is correctly set
    # multivariate = segmenter.get_class_tag(tag_name="capability:multivariate")
    # Test axis is correctly set.
