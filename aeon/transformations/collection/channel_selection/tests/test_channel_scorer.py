"""Test channel scorer channel selector."""

import numpy as np
import pytest

from aeon.regression import DummyRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.mock_estimators import MockClassifierFullTags
from aeon.transformations.collection.channel_selection import ChannelScorer


def test_channel_scorer_with_classifier():
    """Test the channel scorer for classifier."""
    # Test selects the correct number of channels
    cs = ChannelScorer(estimator=MockClassifierFullTags(), proportion=0.5)
    X, _ = make_example_3d_numpy(n_cases=20, n_channels=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    cs.fit(X, y)
    assert len(cs.channels_selected_) == 5
    X, _ = make_example_3d_numpy(n_cases=20, n_channels=9)
    assert len(cs.channels_selected_) == 5

    # Test for proportion out of range
    with pytest.raises(ValueError, match="proportion must be in the range 0-1"):
        cs = ChannelScorer(estimator=MockClassifierFullTags(), proportion=1.1)
        cs.fit(X, y)

    with pytest.raises(ValueError, match="proportion must be in the range 0-1"):
        ChannelScorer(estimator=MockClassifierFullTags(), proportion=-1)
        cs.fit(X, y)

    # Test for selecting all channels when proportion=1.0
    cs = ChannelScorer(estimator=MockClassifierFullTags(), proportion=1.0)
    cs.fit(X, y)
    assert len(cs.channels_selected_) == 9

    # Test for invalid estimator
    with pytest.raises(
        ValueError,
        match="parameter estimator must be an instance of BaseClassifier, "
        "BaseRegressor",
    ):
        cs = ChannelScorer(estimator="FOOBAR", proportion=0.5)
        cs.fit(X, y)


def test_channel_scorer_with_regressor():
    """Test the channel scorer for regressor."""
    # Test selects the correct number of channels
    cs = ChannelScorer(estimator=DummyRegressor(), proportion=0.5)
    X, _ = make_example_3d_numpy(n_cases=20, n_channels=10)
    y = np.random.rand(20)
    cs.fit(X, y)
    assert len(cs.channels_selected_) == 5

    X, _ = make_example_3d_numpy(n_cases=20, n_channels=9)
    cs.fit(X, y)
    assert len(cs.channels_selected_) == 5

    # Test for proportion out of range
    with pytest.raises(ValueError, match="proportion must be in the range 0-1"):
        cs = ChannelScorer(estimator=DummyRegressor(), proportion=1.1)
        cs.fit(X, y)

    with pytest.raises(ValueError, match="proportion must be in the range 0-1"):
        cs = ChannelScorer(estimator=DummyRegressor(), proportion=-0.1)
        cs.fit(X, y)

    # Test for selecting all channels when proportion=1.0
    cs = ChannelScorer(estimator=DummyRegressor(), proportion=1.0)
    cs.fit(X, y)
    assert len(cs.channels_selected_) == 9

    # Test for invalid estimator
    with pytest.raises(
        ValueError,
        match="parameter estimator must be an instance of BaseClassifier, "
        "BaseRegressor",
    ):
        cs = ChannelScorer(estimator="FOOBAR", proportion=0.5)
        cs.fit(X, y)
