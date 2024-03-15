"""Test base segmenter."""

import numpy as np
import pytest

from aeon.segmentation.base import BaseSegmenter
from aeon.testing.mock_estimators import MockSegmenter, SupervisedMockSegmenter


def test_fit_predict_correct():
    """
    Test returns self.

    Test same on two calls same input type.
    """
    x_correct = np.array([0, 0, 0, 1, 1])
    seg = MockSegmenter()
    res = seg.fit(x_correct)
    assert isinstance(res, BaseSegmenter)
    assert res.is_fitted
    assert res.axis == 1
    seg.set_tags(**{"fit_is_empty": True})
    res = seg.fit(x_correct)
    assert res.is_fitted
    res = seg.fit_predict(x_correct)
    assert isinstance(res, np.ndarray)
    seg = SupervisedMockSegmenter()
    res = seg.fit(x_correct, y=x_correct)
    assert res.is_fitted
    with pytest.raises(
        ValueError, match="Tag requires_y is true, but fit called with y=None"
    ):
        seg.fit(x_correct)


def test_to_classification():
    """Test class method to_classification."""
    labels = BaseSegmenter.to_classification([2, 8], length=10)
    assert isinstance(labels, np.ndarray)
    labels = BaseSegmenter.to_classification([2, 8], 10)
    assert np.array_equal(labels, np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0]))
    labels = BaseSegmenter.to_classification([0], 4)
    assert np.array_equal(labels, np.array([1, 0, 0, 0]))
    labels = BaseSegmenter.to_classification([0, 1, 2, 3], 4)
    assert np.array_equal(labels, np.array([1, 1, 1, 1]))


def test_to_clusters():
    """Test class method to_clusters."""
    labels = BaseSegmenter.to_clusters([2, 8], length=10)
    assert isinstance(labels, np.ndarray)
    labels = BaseSegmenter.to_clusters([2, 8], 10)
    assert np.array_equal(labels, np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2]))
    labels = BaseSegmenter.to_clusters([1, 2, 3], 4)
    assert np.array_equal(labels, np.array([0, 1, 2, 3]))
