"""Test tselect channel channel selector."""

import numpy as np
import pytest

from aeon.transformations.collection.channel_selection import TSelect
from aeon.transformations.collection.channel_selection._tselect import (
    _cluster_correlations,
    _extract_single_pass_statistics,
    _interpolate_nan_3d,
    _one_hot_encode,
    _probabilities_to_rank,
    _roc_auc_score_like_tselect,
    _spearman_rank_matrix,
)


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


def test_tselect_fit_with_stratified_validation_split():
    """Test TSelect fits with an imbalanced class-ordered validation split."""
    rng = np.random.RandomState(4)
    X = rng.normal(size=(12, 3, 20))
    y = np.array([0] * 10 + [1] * 2)

    TSelect(validation_size=0.25, random_state=1).fit(X, y)


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


def test_tselect_fit_validates_x_and_y():
    """Test TSelect raises informative errors for malformed inputs."""
    with pytest.raises(ValueError, match="3D numpy array"):
        TSelect()._fit(np.zeros((5, 5)), np.array([0, 1, 0, 1, 0]))

    with pytest.raises(ValueError, match="at least one channel"):
        TSelect()._fit(np.zeros((5, 0, 5)), np.array([0, 1, 0, 1, 0]))

    with pytest.raises(ValueError, match="same number of cases"):
        TSelect()._fit(np.zeros((5, 2, 5)), np.array([0, 1, 0]))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"irrelevant_selector": "x"}, "irrelevant_selector must be a bool"),
        ({"redundant_selector": "x"}, "redundant_selector must be a bool"),
        ({"irrelevant_hard_threshold": 1.5}, "irrelevant_hard_threshold"),
        ({"irrelevant_percentage_to_keep": 1.5}, "irrelevant_percentage_to_keep"),
        ({"redundant_correlation_threshold": 1.5}, "redundant_correlation_threshold"),
        ({"validation_size": 1.5}, "validation_size"),
    ],
)
def test_tselect_validate_parameters_errors(kwargs, match):
    """Test TSelect parameter validation rejects out-of-range values."""
    with pytest.raises(ValueError, match=match):
        TSelect(**kwargs)._validate_parameters()


def test_tselect_restores_channels_when_none_pass_hard_threshold():
    """Test TSelect restores all channels if none pass the hard AUC threshold."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 3, 15))
    y = np.array([0, 1] * 20)

    with pytest.warns(UserWarning, match="No channel passed the hard AUC threshold"):
        selector = TSelect(irrelevant_hard_threshold=1.0, random_state=0)
        selector.fit(X, y)

    assert len(selector.channels_selected_) >= 1
    assert not np.isnan(selector.channel_scores_).any()


def test_tselect_percentage_filter_fallback_warns():
    """Test TSelect warns and falls back when percentage filtering empties ranks."""
    rng = np.random.RandomState(2)
    X = rng.normal(size=(60, 3, 20))
    y = np.array([0, 1] * 30)
    X[y == 1, 0, :] += 2.0

    with pytest.warns(UserWarning, match="No channel passed percentage AUC filtering"):
        selector = TSelect(
            irrelevant_percentage_to_keep=0.0,
            irrelevant_hard_threshold=0.5,
            redundant_correlation_threshold=0.95,
            random_state=0,
        )
        selector.fit(X, y)

    assert len(selector.channels_selected_) >= 1


def test_tselect_irrelevant_selector_false_keeps_all_channels():
    """Test TSelect keeps all channels when irrelevant_selector is False."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 3, 15))
    y = np.array([0, 1] * 20)

    selector = TSelect(
        irrelevant_selector=False,
        redundant_selector=False,
        random_state=0,
    )
    selector.fit(X, y)

    assert selector.channels_selected_ == list(range(3))
    assert selector.removed_series_auc_ == {}


def test_tselect_removes_redundant_channel_in_cluster():
    """Test TSelect removes the worse channel from a correlated cluster."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 2, 20))
    y = np.array([0, 1] * 30)
    X[y == 1, 0, :] += 2.0
    X[:, 1, :] = X[:, 0, :]

    selector = TSelect(
        redundant_correlation_threshold=0.5,
        irrelevant_percentage_to_keep=1.0,
        random_state=0,
    )
    selector.fit(X, y)

    assert len(selector.clusters_) == 1
    assert len(selector.clusters_[0]) == 2
    assert len(selector.removed_series_corr_) == 1


def test_tselect_train_validation_split_explicit_and_large():
    """Test the validation split honours an explicit size and large n_cases."""
    selector = TSelect(validation_size=0.3)
    y = np.array([0, 1] * 25)
    train_idx, valid_idx = selector._train_validation_split(50, y)
    assert len(train_idx) == 35
    assert len(valid_idx) == 15

    selector = TSelect()
    y = np.array([0, 1] * 75)
    train_idx, valid_idx = selector._train_validation_split(150, y)
    assert len(train_idx) + len(valid_idx) == 150
    assert len(train_idx) == 99


def test_tselect_train_validation_split_falls_back_when_stratify_fails():
    """Test the split falls back to unstratified when a class has one member."""
    selector = TSelect(validation_size=0.3)
    y = np.array([0] * 9 + [1])

    with pytest.warns(UserWarning, match="Falling back to an unstratified"):
        train_idx, valid_idx = selector._train_validation_split(10, y)

    assert len(train_idx) + len(valid_idx) == 10


def test_tselect_preprocess_interpolates_nans():
    """Test that NaNs are interpolated away during preprocessing."""
    selector = TSelect()
    X = np.zeros((2, 2, 5))
    X[0, 0, :] = [np.nan, 1.0, 2.0, 3.0, np.nan]

    X_scaled = selector._preprocess(X)

    assert not np.isnan(X_scaled).any()


def test_tselect_prepare_channel_features_drops_all_nan_train_column():
    """Test all-NaN training columns are dropped before scaling."""
    selector = TSelect()
    features = np.array(
        [
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
            [4.0, 5.0],
        ]
    )

    features_train, features_valid = selector._prepare_channel_features(
        features, np.array([0, 1, 2]), np.array([3]), channel=0
    )

    assert features_train.shape[1] == 1
    assert features_valid.shape[1] == 1


def test_tselect_prepare_channel_features_all_nan_raises():
    """Test an entirely NaN channel raises a ValueError."""
    selector = TSelect()
    features = np.array(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [4.0, 5.0],
        ]
    )

    with pytest.raises(ValueError, match="All extracted features for channel 0"):
        selector._prepare_channel_features(
            features, np.array([0, 1, 2]), np.array([3]), channel=0
        )


def test_tselect_prepare_channel_features_replaces_train_nans():
    """Test partial NaNs in training features are replaced by the column mean."""
    selector = TSelect()
    features = np.array(
        [
            [1.0, np.nan],
            [2.0, 4.0],
            [3.0, 5.0],
            [4.0, 5.0],
        ]
    )

    features_train, features_valid = selector._prepare_channel_features(
        features, np.array([0, 1, 2]), np.array([3]), channel=0
    )

    assert not np.isnan(features_train).any()
    assert not np.isnan(features_valid).any()


def test_tselect_filter_auc_percentage_edge_cases():
    """Test percentage filtering short-circuits for empty ranks and n_remove=0."""
    selector = TSelect(irrelevant_percentage_to_keep=1.0)
    selector.channel_scores_ = np.array([0.9, 0.7])
    selector.removed_series_auc_ = {}

    assert selector._filter_auc_percentage({}) == {}

    ranks = {0: np.array([0]), 1: np.array([1])}
    assert selector._filter_auc_percentage(ranks) == ranks


def test_tselect_get_test_params():
    """Test TSelect test parameters are valid and usable."""
    params = TSelect._get_test_params()
    selector = TSelect(**params)
    assert isinstance(selector, TSelect)


def test_extract_single_pass_statistics_edge_cases():
    """Test summary statistics for all-NaN and constant-valued rows."""
    all_nan = _extract_single_pass_statistics(np.array([[np.nan, np.nan, np.nan]]))
    assert all_nan[0, 0] == 0
    assert all_nan[0, 1] == 0
    assert np.isnan(all_nan[0, 2:]).all()

    constant = _extract_single_pass_statistics(np.array([[5.0, 5.0, 5.0]]))
    assert constant[0, 2] == 5.0
    assert constant[0, 5] == 0.0
    assert np.isnan(constant[0, 6:]).all()


def test_interpolate_nan_3d():
    """Test forward/backward filling of NaNs in a 3D array."""
    X = np.zeros((2, 2, 5))
    X[0, 0, :] = [np.nan, np.nan, 1.0, 2.0, np.nan]
    X[1, 1, :] = np.nan

    _interpolate_nan_3d(X)

    assert not np.isnan(X).any()
    np.testing.assert_array_equal(X[0, 0], [1.0, 1.0, 1.0, 2.0, 2.0])
    np.testing.assert_array_equal(X[1, 1], np.zeros(5))


def test_interpolate_nan_3d_fills_internal_gap():
    """Test an internal NaN gap is forward-filled from the preceding value."""
    X = np.zeros((1, 1, 5))
    X[0, 0, :] = [1.0, np.nan, 3.0, np.nan, 5.0]

    _interpolate_nan_3d(X)

    np.testing.assert_array_equal(X[0, 0], [1.0, 1.0, 3.0, 3.0, 5.0])


def test_roc_auc_score_like_tselect_missing_class_raises():
    """Test a missing validation class raises a ValueError."""
    with pytest.raises(ValueError, match="Not all classes are present"):
        _roc_auc_score_like_tselect(
            np.array([0, 0, 0]), np.array([[0.5, 0.5]] * 3), np.array([0, 1])
        )


def test_one_hot_encode_unknown_label_raises():
    """Test an unseen label raises a ValueError."""
    with pytest.raises(ValueError, match="absent from the fitted"):
        _one_hot_encode(np.array([0, 5]), np.array([0, 1]))


def test_probabilities_to_rank_1d():
    """Test 1D probabilities are converted into ranks."""
    ranks = _probabilities_to_rank(np.array([0.1, 0.9, 0.5]))
    np.testing.assert_array_equal(ranks, [3, 1, 2])


def test_spearman_rank_matrix_constant_input_and_2d():
    """Test constant-input correlations are treated as zero, and 2D matrices work."""
    corr = _spearman_rank_matrix(np.array([1, 1, 1]), np.array([1, 2, 3]))
    assert corr == np.array([0.0])

    corr = _spearman_rank_matrix(np.array([1, 2, 3]), np.array([3, 2, 1]))
    assert corr == np.array([-1.0])

    corr_2d = _spearman_rank_matrix(
        np.array([[1, 2], [2, 2], [3, 2]]), np.array([[3, 1], [2, 2], [1, 3]])
    )
    np.testing.assert_array_equal(corr_2d, [-1.0, 0.0])


@pytest.mark.parametrize(
    "rank_correlations,included,expected",
    [
        ({(0, 1): np.array([0.9])}, {0, 1}, [[0, 1]]),
        ({(0, 1): np.array([0.1])}, {0, 1}, [[0], [1]]),
        ({(0, 2): np.array([0.9])}, {0, 1}, [[0], [1]]),
    ],
)
def test_cluster_correlations_basic(rank_correlations, included, expected):
    """Test basic clustering branches: merge, below-threshold, not-included."""
    clusters = _cluster_correlations(rank_correlations, included, threshold=0.7)
    assert clusters == expected


def test_cluster_correlations_merges_separate_clusters():
    """Test two correlated clusters are fully merged when no split is possible."""
    rank_correlations = {
        (0, 1): np.array([0.9]),
        (2, 3): np.array([0.9]),
        (1, 2): np.array([0.9]),
    }
    clusters = _cluster_correlations(rank_correlations, {0, 1, 2, 3}, threshold=0.7)
    assert clusters == [[0, 1, 2, 3]]


def test_cluster_correlations_splits_partially_correlated_cluster():
    """Test a partially correlated channel splits an existing cluster."""
    rank_correlations = {
        (0, 1): np.array([0.9]),
        (0, 2): np.array([0.1]),
        (1, 2): np.array([0.9]),
    }
    clusters = _cluster_correlations(rank_correlations, {0, 1, 2}, threshold=0.7)
    assert clusters == [[2, 1], [0]]


def test_cluster_correlations_appends_when_fully_correlated():
    """Test a channel is appended directly when correlated with a whole cluster."""
    rank_correlations = {
        (0, 1): np.array([0.9]),
        (0, 2): np.array([0.9]),
        (1, 2): np.array([0.9]),
    }
    clusters = _cluster_correlations(rank_correlations, {0, 1, 2}, threshold=0.7)
    assert clusters == [[0, 1, 2]]


def test_cluster_correlations_transform_finds_uncorrelated_pair():
    """Test merging two clusters that contain an uncorrelated cross pair."""
    rank_correlations = {
        (0, 1): np.array([0.9]),
        (2, 3): np.array([0.9]),
        (1, 2): np.array([0.9]),
        (0, 3): np.array([0.1]),
    }
    clusters = _cluster_correlations(rank_correlations, {0, 1, 2, 3}, threshold=0.7)
    assert clusters == [[3, 2, 1], [0]]


def test_cluster_correlations_split_produces_multi_channel_groups():
    """Test a split where both resulting groups end up containing channels."""
    rank_correlations = {
        (0, 1): np.array([0.9]),
        (1, 2): np.array([0.9]),
        (2, 3): np.array([0.1]),
        (0, 2): np.array([1.0]),
        (0, 3): np.array([0.71]),
        (1, 3): np.array([0.71]),
    }
    clusters = _cluster_correlations(rank_correlations, {0, 1, 2, 3}, threshold=0.7)
    assert clusters == [[2, 0, 1], [3]]
