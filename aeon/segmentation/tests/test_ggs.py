"""Tests for _ggs module."""

import numpy as np
import pytest

from aeon.segmentation import GreedyGaussianSegmenter
from aeon.segmentation._ggs import _GGS


@pytest.fixture
def univariate_mean_shift():
    """Generate simple mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return x[:, np.newaxis]


def test_GGS_find_change_points(univariate_mean_shift):
    """Test the _GGS core estimator."""
    ggs = _GGS(k_max=10, lamb=1.0)
    pred = ggs.find_change_points(univariate_mean_shift)
    assert isinstance(pred, list)
    assert len(pred) == 5


def test_GreedyGaussianSegmentation(univariate_mean_shift):
    """Test the GreedyGaussianSegmentation."""
    ggs = GreedyGaussianSegmenter(k_max=5, lamb=0.5)
    assert ggs.get_params() == {
        "k_max": 5,
        "lamb": 0.5,
        "verbose": False,
        "max_shuffles": 250,
        "random_state": None,
    }
