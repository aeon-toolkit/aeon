"""Functions to compute thresholds to convert anomaly scores to binary predictions."""

from __future__ import annotations

__maintainer__ = ["SebastianSchmidl"]
__all__ = [
    "percentile_threshold",
    "sigma_threshold",
    "top_k_points_threshold",
    "top_k_ranges_threshold",
]


import numpy as np


def percentile_threshold(y_score: np.ndarray, percentile: int) -> float:
    """Calculate a threshold based on a percentile of the anomaly scores.

    Uses the xth-percentile of the anomaly scoring as threshold ignoring NaNs and using
    a linear interpolation.

    Parameters
    ----------
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    percentile : int
        Percentile to use as threshold between 0 and 100.

    Returns
    -------
    float
        Threshold based on the percentile.
    """
    return np.nanpercentile(y_score, percentile)


def sigma_threshold(y_score: np.ndarray, factor: float = 2) -> float:
    r"""Calculate a threshold based on the standard deviation of the anomaly scores.

    Computes a threshold :math:``\theta`` based on the anomaly scoring's mean
    :math:``\mu_s`` and the standard deviation :math:``\sigma_s``, ignoring NaNs:

    .. math::
       \theta = \mu_{s} + x \cdot \sigma_{s}

    Parameters
    ----------
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    factor : float
        Number of standard deviations to use as threshold (:math:``x``).

    Returns
    -------
    float
        Threshold based on the standard deviation.
    """
    return np.nanmean(y_score) + factor * np.nanstd(y_score)


def top_k_points_threshold(
    y_true: np.ndarray, y_score: np.ndarray, k: int | None = None
) -> float:
    """Calculate a threshold such that at least ``k`` anomalous points are found.

    The anomalies are single-point anomalies.

    Computes a threshold based on the number of expected anomalies (number of
    anomalies). This method iterates over all possible thresholds from high to low to
    find the first threshold that yields ``k`` or more anomalous points. If ``k``
    is ``None``,the ground truth data is used to calculate the real number of
    anomalies.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    k : optional int
        Number of expected anomalies. If ``k`` is ``None``, the ground truth data
        is used to calculate the real number of anomalies.

    Returns
    -------
    float
        Threshold such that there are at least ``k`` anomalous points.
    """
    if k is None:
        return np.nanpercentile(y_score, (1 - y_true.sum() / y_true.shape[0]) * 100)
    else:
        return np.nanpercentile(y_score, (1 - k / y_true.shape[0]) * 100)


def top_k_ranges_threshold(
    y_true: np.ndarray, y_score: np.ndarray, k: int | None = None
) -> float:
    """Calculate a threshold such that at least ``k`` anomalies are found.

    The anomalies are either single-points anomalies or continuous anomalous ranges.

    Computes a threshold based on the number of expected anomalous subsequences /
    ranges (number of anomalies). This method iterates over all possible thresholds
    from high to low to find the first threshold that yields `k` or more continuous
    anomalous ranges. If ``k`` is ``None``, the ground truth data is used to
    calculate the real number of anomalies (anomalous ranges).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    k : optional int
        Number of expected anomalies. If ``k`` is ``None``, the ground truth data
        is used to calculate the real number of anomalies.

    Returns
    -------
    float
        Threshold such that there are at least ``k`` anomalous ranges.
    """
    if k is None:
        k = _count_anomaly_ranges(y_true)
    thresholds = np.unique(y_score)[::-1]

    # exclude minimum from thresholds, because all points are >= minimum!
    for t in thresholds[:-1]:
        y_pred = np.array(y_score >= t, dtype=np.int_)
        detected_n = _count_anomaly_ranges(y_pred)
        if detected_n >= k:
            return t


def _count_anomaly_ranges(y: np.ndarray) -> int:
    """Count the number of continuous anomalous ranges in a binary sequence.

    Parameters
    ----------
    y : np.ndarray
        Binary sequence of shape (n_instances,).

    Returns
    -------
    int
        Number of continuous anomalous ranges.
    """
    return int(np.sum(np.diff(np.r_[0, y, 0]) == 1))
