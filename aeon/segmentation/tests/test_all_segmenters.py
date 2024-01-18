"""Tests for all segmenters."""
import numpy as np
import pytest

from aeon.base._base_series import VALID_INNER_TYPES
from aeon.registry import all_estimators

ALL_SEGMENTERS = all_estimators(
    estimator_types="segmenter",
    return_names=False,
)


@pytest.mark.parametrize("segmenter", ALL_SEGMENTERS)
def test_segmenter_base_functionality(segmenter):
    """Test does not override final methods."""
    assert "fit" not in segmenter.__dict__
    assert "predict" not in segmenter.__dict__
    # Test that all segmenters implement abstract predict.
    assert "_predict" in segmenter.__dict__
    # Test that fit_is_empty is correctly set
    fit_is_empty = segmenter.get_class_tag(tag_name="fit_is_empty")
    assert not fit_is_empty == "_fit" not in segmenter.__dict__
    # Test valid tag for X_inner_type
    X_inner_type = segmenter.get_class_tag(tag_name="X_inner_type")
    assert X_inner_type in VALID_INNER_TYPES


@pytest.mark.parametrize("segmenter", ALL_SEGMENTERS)
def test_segmenter_instance(segmenter):
    instance = segmenter.create_test_instance()
    multivariate = segmenter.get_class_tag(tag_name="capability:multivariate")
    X = np.random.random(size=(5, 20))
    # Also test does not fail if y is passed
    y = np.array([0, 0, 0, 1, 1])
    # Test that capability:multivariate is correctly set
    if multivariate:
        instance.fit_predict(X, y)
    else:
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            instance.fit_predict(X, y)
    # Test that output is correct type
    X = np.random.random(size=(20))
    output = instance.fit_predict(X)
    assert isinstance(output, list)
    # Test output is consistent with input
    assert max(output) < len(X)
    assert min(output) > 0
    # Test in ascending order
    assert all(output[i] <= output[i + 1] for i in range(len(output) - 1))
    dense = segmenter.get_class_tag(tag_name="returns_dense")
    if dense:
        assert len(output) == len(X)
    else:
        assert len(output) < len(X)
