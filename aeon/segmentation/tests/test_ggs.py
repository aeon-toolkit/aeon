"""Tests for _ggs module."""

from copy import deepcopy

import numpy as np
import pytest

from aeon.segmentation import GreedyGaussianSegmenter
from aeon.segmentation._ggs import _GGS
from aeon.testing.utils.deep_equals import deep_equals


@pytest.fixture
def univariate_mean_shift():
    """Generate simple mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return x[:, np.newaxis]


@pytest.fixture
def multivariate_mean_shift():
    """Generate simple multivariate mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return np.vstack((x, x.max() - x)).T


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


def test_GreedyGaussianSegmenter_predict_does_not_change_state(
    multivariate_mean_shift,
):
    """Test predict does not mutate the inner _GGS object.

    Regression test for #3823: predict-type methods must not change the
    estimator's attributes.
    """
    ggs = GreedyGaussianSegmenter(
        k_max=3, lamb=0.5, max_shuffles=5, random_state=1
    ).fit(multivariate_mean_shift)

    state_before = deepcopy(ggs.__dict__)
    labels_first = ggs.predict(multivariate_mean_shift)
    labels_second = ggs.predict(multivariate_mean_shift)

    assert deep_equals(state_before, ggs.__dict__)
    assert not hasattr(ggs, "ggs")
    # repeated calls give the same segmentation
    assert np.array_equal(labels_first, labels_second)


def test_GreedyGaussianSegmenter_set_params_reaches_algorithm(
    multivariate_mean_shift,
):
    """Test set_params updates the parameters used for segmentation.

    Regression test for #3823: the _GGS work object was built in __init__,
    so set_params never reached the algorithm actually used.
    """
    ggs = GreedyGaussianSegmenter(
        k_max=5, lamb=0.5, max_shuffles=5, random_state=1
    ).fit(multivariate_mean_shift)
    ggs.set_params(k_max=1)

    assert ggs._get_ggs().k_max == 1
    labels = ggs.predict(multivariate_mean_shift)
    # at most k_max change points -> at most k_max + 1 segments
    assert len(np.unique(labels)) <= 2
