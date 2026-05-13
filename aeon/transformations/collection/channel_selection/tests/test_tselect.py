"""Test tselect channel channel selector."""

import numpy as np

from aeon.transformations.collection.channel_selection import TSelect


def test_tselect_fit_transform_shape():
    """Test tselect channel selector."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(30, 5, 20))
    y = np.array([0, 1] * 15)

    selector = TSelect(
        irrelevant_percentage_to_keep=0.4,
        redundant_correlation_threshold=0.7,
        random_state=0,
    )
    Xt = selector.fit_transform(X, y)

    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[2] == X.shape[2]
    assert 1 <= Xt.shape[1] <= X.shape[1]
    assert len(selector.channels_selected_) == Xt.shape[1]


def test_tselect_attributes():
    """Test tselect channel selector."""
    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, 4, 15))
    y = np.array([0, 1] * 20)

    selector = TSelect(random_state=0)
    selector.fit(X, y)

    assert selector.channel_scores_.shape == (4,)
    assert selector.channel_correlations_.shape == (4, 4)
    assert isinstance(selector.clusters_, list)
    assert all(isinstance(ch, int) for ch in selector.channels_selected_)


def test_tselect_keeps_predictive_channel():
    """Test t-select channel selector."""
    rng = np.random.RandomState(2)
    X = rng.normal(size=(60, 3, 20))
    y = np.array([0, 1] * 30)

    X[y == 1, 0, :] += 2.0

    selector = TSelect(
        irrelevant_percentage_to_keep=0.0,
        redundant_correlation_threshold=0.95,
        random_state=0,
    )
    selector.fit(X, y)

    assert 0 in selector.channels_selected_
