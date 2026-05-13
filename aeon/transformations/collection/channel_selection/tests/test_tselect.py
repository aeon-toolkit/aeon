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


def test_tselect_default_validation_split_with_100_cases():
    """Test TSelect default validation split at the 100-case boundary."""
    rng = np.random.RandomState(3)
    X = rng.normal(size=(100, 3, 20))
    y = np.array([0, 1] * 50)

    selector = TSelect(random_state=0)
    selector.fit(X, y)

    assert len(selector.channels_selected_) >= 1


def test_tselect_zero_percentage_keeps_hard_threshold_channels():
    """Test percentage-filter fallback keeps only hard-threshold channels."""
    rng = np.random.RandomState(2)
    X = rng.normal(size=(60, 3, 20))
    y = np.array([0, 1] * 30)

    X[y == 1, 0, :] += 2.0

    selector = TSelect(
        irrelevant_percentage_to_keep=0.0,
        irrelevant_hard_threshold=0.5,
        redundant_correlation_threshold=0.95,
        random_state=0,
    )
    selector.fit(X, y)

    hard_threshold_channels = np.flatnonzero(selector.channel_scores_ >= 0.5)
    assert set(selector.channels_selected_).issubset(set(hard_threshold_channels))


def test_tselect_percentage_filter_removes_cutoff_ties():
    """Test percentage filtering follows TSelect cutoff-threshold semantics."""
    selector = TSelect(irrelevant_percentage_to_keep=0.5)
    selector.channel_scores_ = np.array([0.9, 0.7, 0.7, 0.6])
    selector.removed_series_auc_ = {}

    filtered = selector._filter_auc_percentage(
        {
            0: np.array([0]),
            1: np.array([1]),
            2: np.array([2]),
            3: np.array([3]),
        }
    )

    assert list(filtered.keys()) == [0]
    assert selector.removed_series_auc_ == {1: 0.7, 2: 0.7, 3: 0.6}


def test_tselect_replaces_validation_only_nans():
    """Test validation-only NaN features are replaced before prediction."""
    selector = TSelect()

    features = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, np.nan],
        ]
    )

    features_train, features_valid = selector._prepare_channel_features(
        features,
        np.array([0, 1]),
        np.array([2, 3]),
        channel=0,
    )

    assert not np.isnan(features_train).any()
    assert not np.isnan(features_valid).any()
