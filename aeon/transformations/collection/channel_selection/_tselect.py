"""TSelect channel selection."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["TSelect"]

import warnings
from math import ceil

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from aeon.transformations.collection.channel_selection.base import BaseChannelSelector


class TSelect(BaseChannelSelector):
    """Select relevant and non-redundant channels using TSelect.

    TSelect first trains a simple classifier on summary statistics extracted
    independently from each channel. Channels with weak validation ROC AUC are
    removed. Remaining channels are ranked by their class-probability outputs
    and redundant channels are grouped using Spearman rank correlation.

    Parameters
    ----------
    irrelevant_selector : bool, default=True
        Whether to remove channels with low individual predictive performance.
    irrelevant_hard_threshold : float, default=0.5
        Hard ROC AUC threshold. Channels below this threshold are removed before
        percentage filtering. If no channel passes the threshold, all channels
        are restored before percentage filtering.
    irrelevant_percentage_to_keep : float, default=0.6
        Proportion of channels to keep after hard-threshold filtering. The best
        channels by ROC AUC are retained.
    redundant_selector : bool, default=True
        Whether to remove redundant channels after relevance filtering.
    redundant_correlation_threshold : float, default=0.7
        Mean absolute Spearman rank correlation threshold used to cluster
        redundant channels.
    validation_size : float or None, default=None
        Size of the validation split. If None, use the TSelect rule: 0.25 for
        datasets with fewer than 100 cases, otherwise use
        ``max(100, round(0.25 * n_cases))`` training cases.
    random_state : int or None, default=0
        Random state used for the validation split and logistic regression.

    Attributes
    ----------
    channels_selected_ : list[int]
        Selected channel indices.
    channel_scores_ : np.ndarray of shape (n_channels,)
        Validation ROC AUC score for each channel.
    channel_correlations_ : np.ndarray of shape (n_channels, n_channels)
        Mean absolute Spearman rank correlation matrix. Entries not computed
        are set to ``np.nan``.
    clusters_ : list[list[int]]
        Clusters of relevant channels before selecting the best channel from
        each cluster.
    estimators_ : list
        Fitted per-channel logistic regression models.
    feature_scalers_ : list
        Fitted per-channel feature scalers.
    dropped_nan_columns_ : list
        Boolean masks identifying all-NaN feature columns dropped per channel.
    """

    _tags = {
        "capability:multivariate": True,
        "requires_y": True,
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        irrelevant_selector=True,
        irrelevant_hard_threshold=0.5,
        irrelevant_percentage_to_keep=0.6,
        redundant_selector=True,
        redundant_correlation_threshold=0.7,
        validation_size=None,
        random_state=0,
    ):
        self.irrelevant_selector = irrelevant_selector
        self.irrelevant_hard_threshold = irrelevant_hard_threshold
        self.irrelevant_percentage_to_keep = irrelevant_percentage_to_keep
        self.redundant_selector = redundant_selector
        self.redundant_correlation_threshold = redundant_correlation_threshold
        self.validation_size = validation_size
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        """Fit TSelect."""
        self._validate_parameters()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(
                "TSelect expects X to be a 3D numpy array with shape "
                "(n_cases, n_channels, n_timepoints)."
            )

        n_cases, n_channels, _ = X.shape
        if n_channels < 1:
            raise ValueError("X must contain at least one channel.")
        if len(y) != n_cases:
            raise ValueError("y must have the same number of cases as X.")

        X = self._preprocess(X)
        train_idx, valid_idx = self._train_validation_split(n_cases)

        self.channel_scores_ = np.full(n_channels, np.nan)
        self.channel_correlations_ = np.full((n_channels, n_channels), np.nan)
        np.fill_diagonal(self.channel_correlations_, 1.0)

        self.estimators_ = [None] * n_channels
        self.feature_scalers_ = [None] * n_channels
        self.dropped_nan_columns_ = [None] * n_channels
        self.rank_correlations_ = {}
        self.removed_series_auc_ = set()
        self.removed_series_corr_ = set()

        ranks = {}
        probabilities = {}

        y_train = y[train_idx]
        y_valid = y[valid_idx]
        highest_removed_auc = -np.inf
        removed_predictions = {}

        for channel in range(n_channels):
            features = _extract_single_pass_statistics(X[:, channel, :])
            features_train, features_valid = self._prepare_channel_features(
                features, train_idx, valid_idx, channel
            )

            clf = LogisticRegression(random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(features_train, y_train)

            proba = clf.predict_proba(features_valid)
            auc = _roc_auc_score_like_tselect(y_valid, proba, clf.classes_)

            self.estimators_[channel] = clf
            self.channel_scores_[channel] = auc
            probabilities[channel] = proba

            if self.irrelevant_selector and auc < self.irrelevant_hard_threshold:
                self.removed_series_auc_.add((channel, auc))
                removed_predictions[channel] = proba
                highest_removed_auc = max(highest_removed_auc, auc)
                continue

            ranks[channel] = _probabilities_to_rank(proba)

        if len(ranks) == 0:
            warnings.warn(
                "No channel passed the hard AUC threshold. Restoring all "
                "channels before percentage filtering. The highest removed "
                f"AUC was {highest_removed_auc}.",
                stacklevel=2,
            )

            for channel, auc in self.removed_series_auc_:
                ranks[channel] = _probabilities_to_rank(removed_predictions[channel])
                self.channel_scores_[channel] = auc

            self.removed_series_auc_ = set()

        if self.irrelevant_selector:
            ranks = self._filter_auc_percentage(ranks)
            if len(ranks) == 0:
                warnings.warn(
                    "No channel passed percentage AUC filtering. Keeping the "
                    "channels that passed the hard AUC threshold.",
                    stacklevel=2,
                )
                ranks = {
                    channel: _probabilities_to_rank(probabilities[channel])
                    for channel in range(n_channels)
                    if not np.isnan(self.channel_scores_[channel])
                }

        if len(ranks) == 0:
            raise RuntimeError("TSelect failed to retain any channels.")

        if not self.redundant_selector or len(ranks) == 1:
            self.clusters_ = [[int(channel)] for channel in ranks.keys()]
            self.channels_selected_ = [int(channel) for channel in ranks.keys()]
            return self

        self.rank_correlations_, included_channels = _pairwise_rank_correlation(ranks)
        self._store_channel_correlations(self.rank_correlations_)

        self.clusters_ = _cluster_correlations(
            self.rank_correlations_,
            included_channels,
            threshold=self.redundant_correlation_threshold,
        )

        self.channels_selected_ = self._choose_from_clusters()
        return self

    def _validate_parameters(self):
        if not isinstance(self.irrelevant_selector, bool):
            raise ValueError("irrelevant_selector must be a bool.")
        if not isinstance(self.redundant_selector, bool):
            raise ValueError("redundant_selector must be a bool.")
        if not 0 <= self.irrelevant_hard_threshold <= 1:
            raise ValueError("irrelevant_hard_threshold must be in [0, 1].")
        if not 0 <= self.irrelevant_percentage_to_keep <= 1:
            raise ValueError("irrelevant_percentage_to_keep must be in [0, 1].")
        if not 0 <= self.redundant_correlation_threshold <= 1:
            raise ValueError("redundant_correlation_threshold must be in [0, 1].")
        if self.validation_size is not None and not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be None or a float in (0, 1).")

    def _preprocess(self, X):
        """Apply TSelect-style channel-wise min-max scaling and NaN filling."""
        self.channel_min_ = np.nanmin(X, axis=(0, 2))
        self.channel_max_ = np.nanmax(X, axis=(0, 2))

        with np.errstate(divide="ignore", invalid="ignore"):
            X_scaled = (X - self.channel_min_[None, :, None]) / (
                self.channel_max_[None, :, None] - self.channel_min_[None, :, None]
            )

        if np.isnan(X_scaled).any():
            _interpolate_nan_3d(X_scaled)

        return X_scaled

    def _train_validation_split(self, n_cases):
        if self.validation_size is not None:
            test_size = self.validation_size
        elif n_cases < 100:
            test_size = 0.25
        else:
            n_train = max(100, round(0.25 * n_cases))
            test_size = 1 - (n_train / n_cases)

        train_idx, valid_idx = train_test_split(
            list(range(n_cases)),
            test_size=test_size,
            random_state=self.random_state,
        )

        return np.asarray(train_idx), np.asarray(valid_idx)

    def _prepare_channel_features(self, features, train_idx, valid_idx, channel):
        features_train = features[train_idx]
        features_valid = features[valid_idx]

        if np.isnan(features_train).any():
            nan_columns = np.isnan(features_train).all(axis=0)
            self.dropped_nan_columns_[channel] = nan_columns
            features_train = features_train[:, ~nan_columns]
            features_valid = features_valid[:, ~nan_columns]
        else:
            self.dropped_nan_columns_[channel] = np.zeros(
                features_train.shape[1], dtype=bool
            )

        if features_train.shape[1] == 0:
            raise ValueError(f"All extracted features for channel {channel} were NaN.")

        if np.isnan(features_train).any():
            _replace_nans_by_column_mean(features_train)
            _replace_nans_by_column_mean(features_valid)

        scaler = MinMaxScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_train = scaler.fit_transform(features_train)
            features_valid = scaler.transform(features_valid)

        self.feature_scalers_[channel] = scaler
        return features_train, features_valid

    def _filter_auc_percentage(self, ranks):
        if not ranks:
            return ranks

        sorted_channels = sorted(
            ranks.keys(),
            key=lambda channel: self.channel_scores_[channel],
            reverse=True,
        )

        n_keep = ceil(len(sorted_channels) * self.irrelevant_percentage_to_keep)
        if n_keep == 0:
            return {}

        kept = sorted_channels[:n_keep]
        removed = sorted_channels[n_keep:]

        for channel in removed:
            self.removed_series_auc_.add((channel, self.channel_scores_[channel]))

        return {channel: ranks[channel] for channel in kept}

    def _store_channel_correlations(self, rank_correlations):
        for (channel_a, channel_b), corr in rank_correlations.items():
            mean_corr = float(np.mean(np.abs(corr)))
            self.channel_correlations_[channel_a, channel_b] = mean_corr
            self.channel_correlations_[channel_b, channel_a] = mean_corr

    def _choose_from_clusters(self):
        selected = []

        for cluster in self.clusters_:
            cluster = list(cluster)
            scores = [self.channel_scores_[channel] for channel in cluster]
            best_idx = int(np.argmax(scores))
            selected_channel = int(cluster[best_idx])
            selected.append(selected_channel)

            for channel in cluster:
                if channel != selected_channel:
                    self.removed_series_corr_.add(channel)

        return selected

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for aeon estimator checks."""
        return {
            "irrelevant_percentage_to_keep": 0.75,
            "redundant_correlation_threshold": 0.8,
            "random_state": 0,
        }


def _extract_single_pass_statistics(X):
    """Extract simple per-case statistics for one channel.

    This mirrors the default TSelect idea of using cheap single-pass summary
    features while avoiding the external TSFuse dependency.
    """
    X = np.asarray(X, dtype=float)

    minimum = np.nanmin(X, axis=1)
    maximum = np.nanmax(X, axis=1)
    mean = np.nanmean(X, axis=1)
    var = np.nanvar(X, axis=1)

    std = np.sqrt(var)
    safe_std = np.where(std > 0, std, 1.0)
    centred = X - mean[:, None]
    z = centred / safe_std[:, None]

    skewness = np.nanmean(z**3, axis=1)
    kurtosis = np.nanmean(z**4, axis=1) - 3.0

    skewness = np.where(std > 0, skewness, 0.0)
    kurtosis = np.where(std > 0, kurtosis, 0.0)

    features = np.column_stack([minimum, maximum, mean, var, skewness, kurtosis])

    return features


def _interpolate_nan_3d(X):
    """Forward fill, then backward fill, NaNs in a 3D array in place."""
    nan_channels = np.isnan(X).any(axis=0).any(axis=1)

    for channel in np.flatnonzero(nan_channels):
        for case in range(X.shape[0]):
            series = X[case, channel]

            if not np.isnan(series).any():
                continue

            if np.isnan(series).all():
                series[:] = 0.0
                continue

            valid = np.flatnonzero(~np.isnan(series))
            first_valid = valid[0]
            last_valid = valid[-1]

            series[:first_valid] = series[first_valid]
            series[last_valid + 1 :] = series[last_valid]

            for i in range(first_valid + 1, last_valid + 1):
                if np.isnan(series[i]):
                    series[i] = series[i - 1]


def _replace_nans_by_column_mean(X):
    """Replace NaNs in a 2D feature matrix with column means in place."""
    column_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(column_mean, inds[1])


def _roc_auc_score_like_tselect(y_true, proba, classes):
    """Compute ROC AUC using one-hot labels, matching upstream TSelect."""
    present = np.unique(y_true)
    if present.shape[0] != proba.shape[1]:
        raise ValueError(
            "Not all classes are present in the validation set. Increase "
            "validation_size to compute the AUC."
        )

    y_onehot = _one_hot_encode(y_true, classes)
    return roc_auc_score(y_onehot, proba)


def _one_hot_encode(y, classes):
    """One-hot encode y using the class order of the fitted classifier."""
    class_to_index = {label: i for i, label in enumerate(classes)}
    y_onehot = np.zeros((len(y), len(classes)), dtype=int)

    for i, label in enumerate(y):
        if label not in class_to_index:
            raise ValueError(
                "Validation labels contain a class absent from the fitted "
                "per-channel classifier."
            )
        y_onehot[i, class_to_index[label]] = 1

    return y_onehot


def _probabilities_to_rank(probabilities):
    """Convert probabilities to ranks, with rank 1 as the highest probability."""
    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 1:
        return _probabilities_to_rank_1d(probabilities)

    ranks = np.empty(probabilities.shape, dtype=int)
    for class_idx in range(probabilities.shape[1]):
        ranks[:, class_idx] = _probabilities_to_rank_1d(probabilities[:, class_idx])

    return ranks


def _probabilities_to_rank_1d(probabilities):
    """Convert a 1D probability vector to ranks."""
    indices = np.flip(np.argsort(probabilities))
    ranks = np.empty(probabilities.shape, dtype=int)

    for rank in range(1, len(probabilities) + 1):
        ranks[indices[rank - 1]] = rank

    return ranks


def _pairwise_rank_correlation(ranks):
    """Compute pairwise Spearman rank correlations between channels."""
    channels = list(ranks.keys())
    rank_correlations = {}

    for i, channel_i in enumerate(channels):
        for channel_j in channels[i + 1 :]:
            rank_correlations[(channel_i, channel_j)] = _spearman_rank_matrix(
                ranks[channel_i],
                ranks[channel_j],
            )

    return rank_correlations, set(channels)


def _spearman_rank_matrix(rank_a, rank_b):
    """Compute class-wise Spearman correlations for two rank matrices."""
    if rank_a.ndim == 1:
        corr = spearmanr(rank_a, rank_b).statistic
        if np.isnan(corr):
            corr = 0.0
        return np.array([corr])

    corrs = np.empty(rank_a.shape[1])
    for class_idx in range(rank_a.shape[1]):
        corr = spearmanr(rank_a[:, class_idx], rank_b[:, class_idx]).statistic
        if np.isnan(corr):
            corr = 0.0
        corrs[class_idx] = corr

    return corrs


def _cluster_correlations(rank_correlations, included_channels, threshold=0.7):
    """Cluster channels using the TSelect non-transitive cluster logic."""
    clusters = []
    unallocated = set(included_channels)

    for channel_a, channel_b in rank_correlations.keys():
        if channel_a in included_channels and channel_b not in included_channels:
            continue

        corr = np.mean(np.abs(rank_correlations[(channel_a, channel_b)]))

        if corr < threshold:
            continue

        if channel_a in unallocated and channel_b in unallocated:
            clusters.append([channel_a, channel_b])
            unallocated.remove(channel_a)
            unallocated.remove(channel_b)

        elif channel_a not in unallocated and channel_b not in unallocated:
            cluster_a = [cluster for cluster in clusters if channel_a in cluster][0]

            if channel_b in cluster_a:
                continue

            cluster_b = [cluster for cluster in clusters if channel_b in cluster][0]

            new_cluster_a, new_cluster_b = _transform_clusters(
                cluster_a,
                cluster_b,
                rank_correlations,
                threshold,
            )

            clusters.remove(cluster_a)
            clusters.remove(cluster_b)

            if len(new_cluster_a) > 1:
                clusters.append(new_cluster_a)
            else:
                unallocated.update(new_cluster_a)

            if len(new_cluster_b) > 1:
                clusters.append(new_cluster_b)
            else:
                unallocated.update(new_cluster_b)

        else:
            allocated_channel = channel_a if channel_a not in unallocated else channel_b
            other_channel = channel_a if channel_a in unallocated else channel_b

            cluster_idx = [
                i for i, cluster in enumerate(clusters) if allocated_channel in cluster
            ][0]

            correlated, _ = _check_correlated(
                other_channel,
                clusters[cluster_idx],
                rank_correlations,
                threshold,
            )

            if all(correlated):
                clusters[cluster_idx].append(other_channel)
                unallocated.remove(other_channel)
            else:
                new_cluster_a, new_cluster_b = _split_cluster_series(
                    other_channel,
                    clusters[cluster_idx],
                    rank_correlations,
                    correlated,
                )

                del clusters[cluster_idx]

                if len(new_cluster_a) > 1:
                    clusters.append(new_cluster_a)
                    unallocated = unallocated.difference(set(new_cluster_a))
                else:
                    unallocated.update(new_cluster_a)

                if len(new_cluster_b) > 1:
                    clusters.append(new_cluster_b)
                    unallocated = unallocated.difference(set(new_cluster_b))
                else:
                    unallocated.update(new_cluster_b)

    clusters.extend([[channel] for channel in sorted(unallocated)])
    return clusters


def _split_cluster_series(series, to_split, rank_correlations, correlated):
    """Split one cluster when a new series only matches part of it."""
    cluster_a = [channel for i, channel in enumerate(to_split) if not correlated[i]]
    cluster_b = [series]

    _split_cluster_cluster(cluster_a, cluster_b, to_split, rank_correlations)

    if len(cluster_a) + len(cluster_b) != len(to_split) + 1:
        raise RuntimeError("TSelect cluster split produced inconsistent clusters.")

    return cluster_a, cluster_b


def _split_cluster_cluster(cluster_a, cluster_b, to_split, rank_correlations):
    """Greedily split channels between two clusters by mean correlation."""
    for channel in to_split:
        if channel in cluster_a or channel in cluster_b:
            continue

        corr_a = _mean_abs_corr_with_cluster(
            channel,
            cluster_a,
            rank_correlations,
        )
        corr_b = _mean_abs_corr_with_cluster(
            channel,
            cluster_b,
            rank_correlations,
        )

        if corr_a > corr_b:
            cluster_a.append(channel)
        else:
            cluster_b.append(channel)


def _transform_clusters(cluster_a, cluster_b, rank_correlations, threshold):
    """Transform two clusters after a cross-cluster correlation is found."""
    uncorrelated_a, _ = _find_uncorrelated_signals(
        cluster_a,
        cluster_b,
        rank_correlations,
        threshold,
    )

    if uncorrelated_a is None:
        return cluster_a + cluster_b, []

    new_cluster_a = [uncorrelated_a]

    correlated, _ = _check_correlated(
        uncorrelated_a,
        cluster_b,
        rank_correlations,
        threshold,
    )

    new_cluster_b = [
        channel for i, channel in enumerate(cluster_b) if not correlated[i]
    ]

    to_split = [
        channel
        for channel in cluster_b + cluster_a
        if channel not in new_cluster_b and channel != uncorrelated_a
    ]

    _split_cluster_cluster(
        new_cluster_a,
        new_cluster_b,
        to_split,
        rank_correlations,
    )

    if len(cluster_a) + len(cluster_b) != len(new_cluster_a) + len(new_cluster_b):
        raise RuntimeError("TSelect cluster transform produced inconsistent clusters.")

    return new_cluster_a, new_cluster_b


def _find_uncorrelated_signals(cluster_a, cluster_b, rank_correlations, threshold):
    """Find one pair of channels across clusters that is not correlated."""
    for channel_a in cluster_a:
        for channel_b in cluster_b:
            correlated, _ = _check_correlated(
                channel_a,
                [channel_b],
                rank_correlations,
                threshold,
            )
            if not correlated[0]:
                return channel_a, channel_b

    return None, None


def _check_correlated(test_channel, corr_channels, rank_correlations, threshold):
    """Check whether one channel is correlated with a list of channels."""
    result = []
    corrs = []

    for channel in corr_channels:
        key = (test_channel, channel)
        if key not in rank_correlations:
            key = (channel, test_channel)

        if key not in rank_correlations:
            result.append(True)
            continue

        mean_corr = np.mean(np.abs(rank_correlations[key]))

        if mean_corr >= threshold:
            result.append(True)
        else:
            result.append(False)

        corrs.append(mean_corr)

    return result, corrs


def _mean_abs_corr_with_cluster(channel, cluster, rank_correlations):
    """Mean absolute rank correlation between a channel and a cluster."""
    corrs = []

    for other_channel in cluster:
        key = (channel, other_channel)
        if key not in rank_correlations:
            key = (other_channel, channel)

        if key in rank_correlations:
            corrs.append(np.mean(np.abs(rank_correlations[key])))

    if len(corrs) == 0:
        return 0.0

    return float(np.mean(corrs))
