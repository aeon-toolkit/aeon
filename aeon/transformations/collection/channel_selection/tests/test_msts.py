"""Tests for the MSTS channel selector."""

import numpy as np
import pytest

from aeon.transformations.collection.channel_selection import MSTS
from aeon.transformations.collection.channel_selection._msts import _merit_score


def test_msts_merit_score_matches_definition():
    """The merit score uses mean class agreement and pair agreement."""
    class_scores = np.array([0.8, 0.6, 0.2])
    pair_scores = np.eye(3)
    pair_scores[0, 1] = pair_scores[1, 0] = 0.5

    expected = 2 * np.mean([0.8, 0.6]) / np.sqrt(2 + 2 * 0.5)
    assert _merit_score(class_scores, pair_scores, (0, 1)) == pytest.approx(expected)


def test_msts_selects_informative_channel_and_subsets_transform():
    """MSTS retains an informative channel and preserves collection shape."""
    rng = np.random.RandomState(0)
    n_cases, n_channels, n_timepoints = 18, 3, 12
    y = np.array([0, 1] * (n_cases // 2))
    X = rng.normal(scale=0.05, size=(n_cases, n_channels, n_timepoints))
    X[y == 1, 0, :] += 2.0

    selector = MSTS(n_splits=3, random_state=0)
    Xt = selector.fit_transform(X, y)

    assert 0 in selector.channels_selected_
    assert Xt.shape == (
        n_cases,
        len(selector.channels_selected_),
        n_timepoints,
    )
    np.testing.assert_array_equal(Xt, X[:, selector.channels_selected_, :])


def test_msts_stores_prediction_and_agreement_metadata():
    """MSTS exposes training predictions and square agreement matrices."""
    rng = np.random.RandomState(1)
    n_cases, n_channels, n_timepoints = 12, 4, 8
    X = rng.normal(size=(n_cases, n_channels, n_timepoints))
    y = np.array(["a", "b"] * (n_cases // 2))

    selector = MSTS(n_splits=2, random_state=0).fit(X, y)

    assert selector.channel_predictions_.shape == (n_cases, n_channels)
    assert selector.channel_class_scores_.shape == (n_channels,)
    assert selector.channel_pair_scores_.shape == (n_channels, n_channels)
    np.testing.assert_allclose(np.diag(selector.channel_pair_scores_), 1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_splits": 1}, "n_splits must be an integer"),
        ({"n_splits": 3, "n_jobs": 1.5}, "n_jobs"),
    ],
)
def test_msts_validates_parameters(kwargs, message):
    """MSTS rejects invalid cross-validation and threading parameters."""
    with pytest.raises(ValueError, match=message):
        MSTS(**kwargs)._validate_parameters()


def test_msts_requires_enough_cases_per_class():
    """MSTS requires every class to appear in each validation-fold scheme."""
    X = np.zeros((5, 2, 8))
    y = np.array([0, 0, 0, 1, 1])

    with pytest.raises(ValueError, match="Each class must contain"):
        MSTS(n_splits=3).fit(X, y)
