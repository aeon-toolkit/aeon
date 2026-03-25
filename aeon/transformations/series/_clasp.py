"""
ClaSP (Classification Score Profile) Transformer implementation.
"""

__maintainer__ = []
__all__ = ["ClaSPTransformer"]

import warnings

import numpy as np
import numpy.fft as fft
import pandas as pd
from numba import njit, objmode, prange

from aeon.transformations.series.base import BaseSeriesTransformer
from aeon.utils.validation import check_n_jobs


def _sliding_window(X, m: int):
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
def _sliding_mean_std(X, m: int):
    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(X)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(X**2)))
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]
    movmean = segSum / m
    movstd = np.sqrt(np.clip(segSumSq / m - (segSum / m) ** 2, 0, None))
    movstd = np.where(np.abs(movstd) < 0.001, 1, movstd)
    return [movmean, movstd]


@njit(fastmath=True, cache=True, parallel=True)
def _compute_distances_iterative(
    X,
    m: int,
    k: int,
    n_jobs: int = 1,
    slack: float = 0.5,
):
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
                dot_rolled = _sliding_dot_product(X[start : start + m], X)
            else:
                dot_rolled = (
                    np.roll(dot_prev, 1)
                    + X[order + m - 1] * X[m - 1 : n + m]
                    - X[order - 1] * np.roll(X[:n], 1)
                )
                dot_rolled[0] = dot_first[order]

            x_mean = means[order]
            x_std = stds[order]

            dist = 2 * m * (1 - (dot_rolled - m * means * x_mean) / (m * stds * x_std))

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
def _calc_knn_labels(knn_mask, split_idx: int, m: int):
    k_neighbours, n_timepoints = knn_mask.shape

    y_true = np.concatenate(
        (
            np.zeros(split_idx, dtype=np.int64),
            np.ones(n_timepoints - split_idx, dtype=np.int64),
        )
    )

    knn_mask_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = knn_mask[i_neighbor]
        knn_mask_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_mask_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    exclusion_zone = np.arange(split_idx - m, split_idx)
    y_pred[exclusion_zone] = np.ones(m, dtype=np.int64)

    return y_true, y_pred


@njit(fastmath=True, cache=False)
def _binary_f1_score(y_true, y_pred):
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
    y_true = y_true == 1
    desc_score_indices = np.arange(y_score.shape[0])[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate(
        (distinct_value_indices, np.array([y_true.size - 1]))
    )

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

    dx = np.diff(fpr)
    direction = 1 if not np.any(dx < 0) else -1
    return direction * np.trapz(tpr, fpr)


@njit(fastmath=True)
def _calc_profile(m: int, knn_mask, score, exclusion_zone: int):
    n_timepoints = knn_mask.shape[1]
    profile = np.full(shape=n_timepoints, fill_value=np.nan, dtype=np.float64)

    for split_idx in range(exclusion_zone, n_timepoints - exclusion_zone):
        y_true, y_pred = _calc_knn_labels(knn_mask, split_idx, m)
        profile[split_idx] = score(y_true, y_pred)

    return profile


def clasp(
    X,
    m: int,
    k_neighbours: int = 3,
    score=_roc_auc_score,
    interpolate: bool = True,
    exclusion_radius: float = 0.05,
    n_jobs: int = 1,
):
    knn_mask = _compute_distances_iterative(X, m, k_neighbours, n_jobs=n_jobs).T

    n_timepoints = knn_mask.shape[1]
    exclusion_zone = max(m, np.int64(n_timepoints * exclusion_radius))

    profile = _calc_profile(m, knn_mask, score, exclusion_zone)

    if interpolate:
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
    return profile, knn_mask


class ClaSPTransformer(BaseSeriesTransformer):

    def __init__(
        self,
        window_length: int = 10,
        scoring_metric: str = "ROC_AUC",
        exclusion_radius: float = 0.05,
        n_jobs: int = 1,
    ):
        self.window_length = int(window_length)
        self.scoring_metric = scoring_metric
        self.exclusion_radius = exclusion_radius
        self.n_jobs = n_jobs
        super().__init__(axis=0)

    def _transform(self, X, y=None):
        n_jobs = check_n_jobs(self.n_jobs)

        if len(X) - self.window_length < 2 * self.exclusion_radius * len(X):
            warnings.warn("Period-Length is larger than size of the time series")

        if X.dtype != np.float64:
            warnings.warn(f"dtype is {X.dtype}, converting to float64")

        scoring_metric_call = self._check_scoring_metric(self.scoring_metric)
        X = X.flatten().astype(np.float64)

        Xt, _ = clasp(
            X,
            self.window_length,
            score=scoring_metric_call,
            exclusion_radius=self.exclusion_radius,
            n_jobs=n_jobs,
        )

        return Xt

    def _check_scoring_metric(self, scoring_metric: str):
        if scoring_metric == "ROC_AUC":
            return _roc_auc_score
        elif scoring_metric == "F1":
            return _binary_f1_score
        else:
            raise ValueError("Invalid scoring metric")
