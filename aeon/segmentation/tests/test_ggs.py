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


def test_ggs_fit_is_empty():
    """Test that GreedyGaussianSegmenter does not implement redundant _fit."""
    # Ensure _fit was removed from the class
    assert "_fit" not in GreedyGaussianSegmenter.__dict__
    
    # Ensure the tag is correctly True
    assert GreedyGaussianSegmenter.get_class_tag("fit_is_empty") is True

    # Ensure calling .fit() works without error
    X = np.random.RandomState(0).randn(30, 2)
    ggs = GreedyGaussianSegmenter(k_max=2, max_shuffles=5)
    result = ggs.fit(X, axis=0)
    assert result is ggs

