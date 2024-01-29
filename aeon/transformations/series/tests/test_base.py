""""Test the base series transformer."""

import numpy as np

from aeon.segmentation import BaseSegmenter


def test_to_classification():
    labels = BaseSegmenter.to_classification(None, [2, 8], 10)
    assert np.array_equal(labels, np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0]))
    labels = BaseSegmenter.to_classification(None, [0], 4)
    assert np.array_equal(labels, np.array([1, 0, 0, 0]))
    labels = BaseSegmenter.to_classification(None, [0, 1, 2, 3], 4)
    assert np.array_equal(labels, np.array([1, 1, 1, 1]))


def test_to_clusters():
    labels = BaseSegmenter.to_clusters(None, [2, 8], 10)
    assert np.array_equal(labels, np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2]))
    labels = BaseSegmenter.to_clusters(None, [1, 2, 3], 4)
    assert np.array_equal(labels, np.array([0, 1, 2, 3]))
