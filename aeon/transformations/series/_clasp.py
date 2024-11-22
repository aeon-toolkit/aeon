"""
ClaSP (Classification Score Profile) Transformer implementation.

Notes
-----
As described in
@inproceedings{clasp2021,
  title={ClaSP - Time Series Segmentation},
  author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
  booktitle={CIKM},
  year={2021}
}
"""

__maintainer__ = []
__all__ = ["ClaSPTransformer"]

import warnings

import numpy as np
import numpy.fft as fft
import pandas as pd
from numba import njit, objmode, prange

from aeon.transformations.series.base import BaseSeriesTransformer


def _sliding_window(X, m):
    """Return the sliding windows for a time series and a window size.

    Parameters
    ----------
    X : array-like, shape = [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows

    Returns
    -------
    windows : array of shape [n-m+1, m]
        The sliding windows of length over the time series of length n
    """
    shape = X.shape[:-1] + (X.shape[-1] - m + 1, m)
    strides = X.strides + (X.strides[-1],)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


@njit(fastmath=True, cache=True)
def _sliding_dot_product(query, time_series):
    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.concatenate((np.array([0]), time_series))
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.concatenate((np.array([0]), query))
        q_add = 1

    query = query[::-1]

    query = np.concatenate((query, np.zeros(n - m + time_series_add - q_add)))

    trim = m - 1 + time_series_add

    with objmode(dot_product="float64[:]"):
        dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))

    return dot_product[trim:]


@njit(fastmath=True, cache=True)
def _sliding_mean_std(X, m):
    """Return the sliding mean and std for a time series and a window size.

    Parameters
    ----------
    X : array-like, shape [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows

    Returns
    -------
    Tuple (float, float)
        The moving mean and moving std
    """
    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(X)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(X**2)))
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]
    movmean = segSum / m

    # avoid dividing by too small std, like 0
    movstd = np.sqrt(np.clip(segSumSq / m - (segSum / m) ** 2, 0, None))
    movstd = np.where(np.abs(movstd) < 0.001, 1, movstd)
    return [movmean, movstd]


@njit(fastmath=True, cache=True, parallel=True)
def _compute_distances_iterative(X, m, k, n_jobs=1, slack=0.5):
    """Compute kNN indices with dot-product.

    No-loops implementation for a time series, given
    a window size and k neighbours.

    Parameters
    ----------
    X : array-like, shape [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows
    k : int
        The number of nearest neighbors
    n_jobs : int, default=1
        Number of jobs to be used.
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

    Returns
    -------
    knns : array-like, shape = [n-m+1, k], dtype=int
        The knns (offsets!) for each subsequence in X
    """
    n = np.int32(X.shape[0] - m + 1)
    halve_m = int(m * slack)

    knns = np.zeros(shape=(n, k), dtype=np.int64)

    means, stds = _sliding_mean_std(X, m)
    dot_first = _sliding_dot_product(X[:m], X)
    bin_size = X.shape[0] // n_jobs

    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min((idx + 1) * bin_size, X.shape[0] - m + 1)

        dot_prev = None
        for order in np.arange(start, end):
            if order == start:
                # first iteration O(n log n)
                dot_rolled = _sliding_dot_product(X[start : start + m], X)
            else:
                # constant time O(1) operations
                dot_rolled = (
                    np.roll(dot_prev, 1)
                    + X[order + m - 1] * X[m - 1 : n + m]
                    - X[order - 1] * np.roll(X[:n], 1)
                )
                dot_rolled[0] = dot_first[order]

            x_mean = means[order]
            x_std = stds[order]

            dist = 2 * m * (1 - (dot_rolled - m * means * x_mean) / (m * stds * x_std))

            # self-join: exclusion zone
            trivialMatchRange = (
                int(max(0, order - halve_m)),
                int(min(order + halve_m + 1, n)),
            )
            dist[trivialMatchRange[0] : trivialMatchRange[1]] = np.inf
            dot_prev = dot_rolled

            if dist.shape[0] >= k:
                knns[order] = np.argpartition(dist, k)[:k]
            else:
                knns[order] = np.arange(dist.shape[0], dtype=np.int64)

    return knns


@njit(fastmath=True, cache=True)
def _calc_knn_labels(knn_mask, split_idx, m):
    """Compute kNN indices relabeling at a given split index.

    Parameters
    ----------
    knn_mask : array-like, shape = [k, n-m+1], dtype=int
        The knn indices for each subsequence
    split_idx : int
        The split index to use
    m : int
        The window size to generate sliding windows

    Returns
    -------
    Tuple (array-like of shape=[n-m+1], array-like of shape=[n-m+1]):
        True labels and predicted labels
    """
    k_neighbours, n_timepoints = knn_mask.shape

    # create labels for given potential split
    y_true = np.concatenate(
        (
            np.zeros(split_idx, dtype=np.int64),
            np.ones(n_timepoints - split_idx, dtype=np.int64),
        )
    )

    knn_mask_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    # relabel the kNN indices
    for i_neighbor in range(k_neighbours):
        neighbours = knn_mask[i_neighbor]
        knn_mask_labels[i_neighbor] = y_true[neighbours]

    # compute kNN prediction
    ones = np.sum(knn_mask_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    # apply exclusion zone at split point
    exclusion_zone = np.arange(split_idx - m, split_idx)
    # exclusion_zone[exclusion_zone < 0] = 0
    y_pred[exclusion_zone] = np.ones(m, dtype=np.int64)

    return y_true, y_pred


@njit(fastmath=True, cache=False)
def _binary_f1_score(y_true, y_pred):
    """Compute f1-score.

    Parameters
    ----------
    y_true : array-like, shape=[n-m+1], dtype = int
        True integer labels for each subsequence
    y_pred : array-like, shape=[n-m+1], dtype = int
        Predicted integer labels for each subsequence

    Returns
    -------
    F1 : float
        F1-score
    """
    f1_scores = np.zeros(shape=2, dtype=np.float64)

    for label in (0, 1):
        tp = np.sum(np.logical_and(y_true == label, y_pred == label))
        fp = np.sum(np.logical_and(y_true != label, y_pred == label))
        fn = np.sum(np.logical_and(y_true == label, y_pred != label))

        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        f1 = 2 * (pr * re) / (pr + re)
        f1_scores[label] = f1

    return np.mean(f1_scores)


@njit(fastmath=True, cache=True)
def _roc_auc_score(y_score, y_true):
    """Compute roc-auc score.

    Parameters
    ----------
    y_true : array-like, shape=[n-m+1], dtype = int
        True integer labels for each subsequence
    y_pred : array-like, shape=[n-m+1], dtype = int
        Predicted integer labels for each subsequence

    Returns
    -------
    F1 : float
        ROC-AUC-score
    """
    # make y_true a boolean vector
    y_true = y_true == 1

    # sort scores and corresponding truth values (y_true is sorted by design)
    desc_score_indices = np.arange(y_score.shape[0])[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate(
        (distinct_value_indices, np.array([y_true.size - 1]))
    )

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.concatenate((np.array([0]), tps))
    fps = np.concatenate((np.array([0]), fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    if fpr.shape[0] < 2:
        return np.nan

    direction = 1
    dx = np.diff(fpr)

    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return np.nan

    area = direction * np.trapz(tpr, fpr)
    return area


@njit(fastmath=True)
def _calc_profile(m, knn_mask, score, exclusion_zone):
    """Calculate ClaSP profile for the kNN indices and a score.

    Parameters
    ----------
    m : int
        The window size to generate sliding windows
    knn_mask : array-like, shape = [k, n-m+1], dtype=int
        The knn indices
    score : function
        Scoring method used
    exclusion_zone : int
        Exclusion zone

    Returns
    -------
    profile : array-like, shape=[n-m+1], dtype = float
        The ClaSP
    """
    n_timepoints = knn_mask.shape[1]
    profile = np.full(shape=n_timepoints, fill_value=np.nan, dtype=np.float64)

    for split_idx in range(exclusion_zone, n_timepoints - exclusion_zone):
        y_true, y_pred = _calc_knn_labels(knn_mask, split_idx, m)
        profile[split_idx] = score(y_true, y_pred)

    return profile


def clasp(
    X,
    m,
    k_neighbours=3,
    score=_roc_auc_score,
    interpolate=True,
    exclusion_radius=0.05,
    n_jobs=1,
):
    """Calculate ClaSP for a time series and a window size.

    Parameters
    ----------
    X : array-like, shape = [n]
        A single univariate time series of length n
    m : int
        The window size to generate sliding windows
    k_neighbours : int
        The number of knn to use
    score : function
        Scoring method used
    interpolate:
        Interpolate the profile
    exclusion_radius : int
        Blind spot of the profile to the corners
    n_jobs : int
        Number of jobs to be used.

    Returns
    -------
    Tuple (array-like of shape [n], array-like of shape [k_neighbours, n])
        The ClaSP and the knn_mask
    """
    knn_mask = _compute_distances_iterative(X, m, k_neighbours, n_jobs=n_jobs).T

    n_timepoints = knn_mask.shape[1]
    exclusion_zone = max(m, np.int64(n_timepoints * exclusion_radius))

    profile = _calc_profile(m, knn_mask, score, exclusion_zone)

    if interpolate:
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
    return profile, knn_mask


class ClaSPTransformer(BaseSeriesTransformer):
    """ClaSP (Classification Score Profile) Transformer.

    Implementation of the Classification Score Profile of a time series.
    ClaSP hierarchically splits a TS into two parts, where each split point is
    determined by training a binary TS classifier for each possible split point and
    selecting the one with highest accuracy, i.e., the one that is best at identifying
    subsequences to be from either of the partitions.

    Parameters
    ----------
    window_length :       int, default = 10
        size of window for sliding.
    scoring_metric :      string, default = ROC_AUC
        the scoring metric to use in ClaSP - choose from ROC_AUC or F1
    exclusion_radius : int
        Exclusion Radius for change points to be non-trivial matches
    n_jobs : int
        Number of jobs to be used.

    Notes
    -----
    As described in
    @inproceedings{clasp2021,
      title={ClaSP - Time Series Segmentation},
      author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
      booktitle={CIKM},
      year={2021}
    }

    Examples
    --------
    >>> from aeon.transformations.series import ClaSPTransformer
    >>> from aeon.segmentation import find_dominant_window_sizes
    >>> from aeon.datasets import load_electric_devices_segmentation
    >>> X, true_period_size, true_cps = load_electric_devices_segmentation()
    >>> dominant_period_size = find_dominant_window_sizes(X)
    >>> clasp = ClaSPTransformer(window_length=dominant_period_size).fit(X)
    >>> profile = clasp.transform(X)
    """

    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
        "requires_y": False,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        window_length=10,
        scoring_metric="ROC_AUC",
        exclusion_radius=0.05,
        n_jobs=1,
    ):
        self.window_length = int(window_length)
        self.scoring_metric = scoring_metric
        self.exclusion_radius = exclusion_radius
        self.n_jobs = n_jobs
        super().__init__(axis=0)

    def _transform(self, X, y=None):
        """Compute ClaSP.

        Takes as input a single time series dataset and returns the
        Classification Score profile for that single time series.

        Parameters
        ----------
        X : numpy.ndarray
            A univariate time series
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 1D numpy.ndarray
            transformed version of X
            ClaSP of the single time series as output
            with length as (n-window_length+1)
        """
        if len(X) - self.window_length < 2 * self.exclusion_radius * len(X):
            warnings.warn(
                "Period-Length is larger than size of the time series", stacklevel=1
            )

        scoring_metric_call = self._check_scoring_metric(self.scoring_metric)

        X = X.flatten()
        Xt, _ = clasp(
            X,
            self.window_length,
            score=scoring_metric_call,
            exclusion_radius=self.exclusion_radius,
            n_jobs=self.n_jobs,
        )

        return Xt

    def _check_scoring_metric(self, scoring_metric):
        """Check which scoring metric to use.

        Parameters
        ----------
        scoring_metric : string
            Choose from "ROC_AUC" or "F1"

        Returns
        -------
        scoring_metric_call : a callable, keyed by the `scoring_metric` input
            _roc_auc_score, if scoring_metric = "ROC_AUC"
            _binary_f1_score, if scoring_metric = "F1"
        """
        valid_scores = ("ROC_AUC", "F1")

        if scoring_metric not in valid_scores:
            raise ValueError(f"invalid input, please use one of {valid_scores}")

        if scoring_metric == "ROC_AUC":
            return _roc_auc_score
        elif scoring_metric == "F1":
            return _binary_f1_score
