"""Merit Score for Time Series channel selection."""

__maintainer__ = ["aeon developers"]
__all__ = ["MSTS"]

from itertools import combinations
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import StratifiedKFold

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector
from aeon.utils.validation import check_n_jobs


def _merit_score(
    class_scores: np.ndarray,
    pair_scores: np.ndarray,
    channels: tuple[int, ...],
) -> float:
    """Calculate the MSTS merit score for a channel subset."""
    k = len(channels)
    dc = float(np.mean(class_scores[list(channels)]))
    if k == 1:
        return dc

    pair_values = pair_scores[np.ix_(channels, channels)]
    upper_values = pair_values[np.triu_indices(k, 1)]
    dd = float(np.mean(upper_values))
    denominator_squared = k + k * (k - 1) * dd
    denominator = np.sqrt(max(denominator_squared, np.finfo(float).eps))
    return float(k * dc / denominator)


class MSTS(BaseChannelSelector):
    """Select channels using the Merit Score for Time Series (MSTS).

    MSTS evaluates each channel with out-of-fold predictions from a 1-nearest
    neighbour dynamic time warping classifier. It measures channel-to-class and
    channel-to-channel agreement with adjusted mutual information, then uses a
    greedy forward search to select the subset with the highest merit score.

    Parameters
    ----------
    n_splits : int, default=3
        Number of stratified folds used to generate single-channel predictions.
    n_jobs : int, default=1
        Number of parallel jobs used by the 1-NN DTW classifiers.
    random_state : int or None, default=None
        Controls shuffling of the stratified folds. If None, folds are not shuffled.

    Attributes
    ----------
    channels_selected_ : list[int]
        Indices of the selected channels, in their original channel order.
    channel_predictions_ : np.ndarray
        Out-of-fold predictions for each training case and channel.
    channel_class_scores_ : np.ndarray
        Adjusted mutual information between each channel's predictions and ``y``.
    channel_pair_scores_ : np.ndarray
        Pairwise adjusted mutual information between channel predictions.
    selection_history_ : list[tuple[tuple[int, ...], float]]
        Selected subsets and their merit scores at each forward-search step.
    selected_merit_score_ : float
        Merit score of the selected channel subset.

    Notes
    -----
    MSTS is implemented as a channel selector, so it does not fit a final
    classifier. The selected channels are applied by ``transform`` and can be
    passed to a downstream multivariate time-series estimator.

    References
    ----------
    .. [1] Kathirgamanathan, B. and Cunningham, P. "Correlation Based Feature
        Subset Selection for Multivariate Time-Series Data." 2021.
        https://arxiv.org/abs/2112.03705
    .. [2] aeon issue #1481, "Implement Merit Score Function channel selection
        algorithm."
        https://github.com/aeon-toolkit/aeon/issues/1481

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon.transformations.collection.channel_selection import MSTS
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=12, n_channels=3, n_timepoints=10, random_state=0
    ... )
    >>> selector = MSTS(n_splits=2, random_state=0).fit(X, y)
    >>> X_selected = selector.transform(X)
    >>> X_selected.shape[1] <= X.shape[1]
    True
    """

    _tags = {
        "capability:multivariate": True,
        "requires_y": True,
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        n_splits: int = 3,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit channel predictions, agreement scores, and the MSTS subset."""
        self._validate_parameters()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_cases, n_channels, _ = X.shape
        self.n_cases_ = n_cases
        self.n_channels_ = n_channels

        class_counts = np.unique(y, return_counts=True)[1]
        if class_counts.size == 0 or np.min(class_counts) < self.n_splits:
            raise ValueError(
                "Each class must contain at least n_splits cases for MSTS."
            )

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.random_state is not None,
            random_state=self.random_state,
        )
        fold_indices = tuple(splitter.split(np.zeros(n_cases), y))

        predictions = np.empty((n_cases, n_channels), dtype=y.dtype)
        for channel in range(n_channels):
            channel_data = X[:, channel : channel + 1, :]
            for train_index, validation_index in fold_indices:
                classifier = KNeighborsTimeSeriesClassifier(
                    n_neighbors=1,
                    distance="dtw",
                    n_jobs=self._n_jobs,
                )
                classifier.fit(channel_data[train_index], y[train_index])
                predictions[validation_index, channel] = classifier.predict(
                    channel_data[validation_index]
                )

        self.channel_predictions_ = predictions
        self.channel_class_scores_ = np.asarray(
            [
                adjusted_mutual_info_score(y, predictions[:, channel])
                for channel in range(n_channels)
            ],
            dtype=float,
        )
        self.channel_pair_scores_ = np.eye(n_channels, dtype=float)
        for first, second in combinations(range(n_channels), 2):
            score = adjusted_mutual_info_score(
                predictions[:, first], predictions[:, second]
            )
            self.channel_pair_scores_[first, second] = score
            self.channel_pair_scores_[second, first] = score

        selected, history = self._forward_search()
        self.channels_selected_ = list(selected)
        self.selection_history_ = history
        self.selected_merit_score_ = history[-1][1]
        return self

    def _forward_search(
        self,
    ) -> tuple[tuple[int, ...], list[tuple[tuple[int, ...], float]]]:
        """Select a channel subset by greedy forward merit maximisation."""
        if self.n_channels_ == 1:
            subset = (0,)
            return subset, [
                (
                    subset,
                    _merit_score(
                        self.channel_class_scores_, self.channel_pair_scores_, subset
                    ),
                )
            ]

        best_subset = (0, 1)
        best_score = _merit_score(
            self.channel_class_scores_, self.channel_pair_scores_, best_subset
        )
        for subset in combinations(range(self.n_channels_), 2):
            score = _merit_score(
                self.channel_class_scores_, self.channel_pair_scores_, subset
            )
            if score > best_score:
                best_subset = subset
                best_score = score

        history = [(best_subset, best_score)]
        while len(best_subset) < self.n_channels_:
            candidate_subset = None
            candidate_score = best_score
            for channel in range(self.n_channels_):
                if channel in best_subset:
                    continue
                subset = tuple(sorted((*best_subset, channel)))
                score = _merit_score(
                    self.channel_class_scores_, self.channel_pair_scores_, subset
                )
                if score > candidate_score:
                    candidate_subset = subset
                    candidate_score = score
            if candidate_subset is None:
                break
            best_subset = candidate_subset
            best_score = candidate_score
            history.append((best_subset, best_score))

        return best_subset, history

    def _validate_parameters(self):
        """Validate constructor parameters."""
        if not isinstance(self.n_splits, (int, np.integer)) or self.n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        self._n_jobs = check_n_jobs(self.n_jobs)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict[str, Any]:
        """Return a small parameter set for estimator checks."""
        return {"n_splits": 2, "n_jobs": 1, "random_state": 0}
