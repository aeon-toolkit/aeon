"""Tests for igts module."""

from copy import deepcopy

import numpy as np
import pytest

from aeon.segmentation import InformationGainSegmenter, entropy
from aeon.segmentation._igts import _IGTS
from aeon.testing.utils.deep_equals import deep_equals


@pytest.fixture
def multivariate_mean_shift():
    """Generate simple mean shift time series."""
    x = np.concatenate(tuple(np.ones(5) * i**2 for i in range(4)))
    return np.vstack((x, x.max() - x))


def test_entropy():
    """Test entropy function."""
    assert entropy(np.ones((10, 1))) == 0.0


def test_igts_identity():
    """Test identity segmentation."""
    X = np.random.random(12).reshape(4, 3)
    id_change_points = _IGTS().identity(X)
    assert id_change_points == [0, 4]


def test_igts_get_candidates():
    """Test get_candidates function."""
    candidates = _IGTS(step=2).get_candidates(n_samples=10, change_points=[0, 4, 6])
    assert candidates == [2, 8]


def test_IGTS_find_change_points(multivariate_mean_shift):
    """Test the _IGTS core estimator."""
    igts = _IGTS(k_max=3, step=1)
    pred = igts.find_change_points(multivariate_mean_shift)
    assert isinstance(pred, list)
    assert len(pred) == 5


def test_InformationGainSegmenter(multivariate_mean_shift):
    """Test the test_InformationGainSegmenter."""
    igts = InformationGainSegmenter(k_max=3, step=1)
    assert igts.get_params() == {"k_max": 3, "step": 1}
    pred = igts.fit_predict(multivariate_mean_shift)
    assert np.array_equal(
        pred,
        np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=int
        ),
    )


def test_InformationGainSegmenter_predict_does_not_change_state(
    multivariate_mean_shift,
):
    """Test predict does not mutate the inner _IGTS object.

    Regression test for #3822: predict-type methods must not change the
    estimator's attributes.
    """
    igts = InformationGainSegmenter(k_max=3, step=1).fit(multivariate_mean_shift)

    state_before = deepcopy(igts.__dict__)
    pred_first = igts.predict(multivariate_mean_shift)
    pred_second = igts.predict(multivariate_mean_shift)

    assert deep_equals(state_before, igts.__dict__)
    assert not hasattr(igts, "_igts")
    # repeated calls give the same segmentation
    assert np.array_equal(pred_first, pred_second)


def test_InformationGainSegmenter_set_params_reaches_algorithm(
    multivariate_mean_shift,
):
    """Test set_params updates the parameters used for segmentation.

    Regression test for #3822: the _IGTS object was built in __init__,
    so set_params never reached the algorithm actually used.
    """
    igts = InformationGainSegmenter(k_max=3, step=1).fit(multivariate_mean_shift)
    igts.set_params(k_max=1)

    assert igts._get_igts().k_max == 1
    pred = igts.predict(multivariate_mean_shift)
    # a single change point splits the series into two segments
    assert len(np.unique(pred)) == 2
