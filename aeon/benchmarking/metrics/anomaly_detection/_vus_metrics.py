"""VUS and AUC metrics for anomaly detection on time series."""

from __future__ import annotations

__maintainer__ = ["SebastianSchmidl"]
__all__ = [
    "range_pr_auc_score",
    "range_roc_auc_score",
    "range_pr_vus_score",
    "range_roc_vus_score",
    "range_pr_roc_auc_support",
]


import warnings

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection._util import check_y


def _anomaly_bounds(y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corresponds to range_convers_new."""
    # convert to boolean/binary
    labels = y_true > 0
    # deal with start and end of time series
    labels = np.diff(np.r_[0, labels, 0])
    # extract begin and end of anomalous regions
    index = np.arange(0, labels.shape[0])
    starts = index[labels == 1]
    ends = index[labels == -1]
    return starts, ends


def _extend_anomaly_labels(
    y_true: np.ndarray, buffer_size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the anomaly labels with slopes on both ends.

    Also makes the labels continuous instead of binary.
    """
    starts, ends = _anomaly_bounds(y_true)

    if buffer_size is None:
        # per default: set buffer size as median anomaly length:
        buffer_size = int(np.median(ends - starts))

    if buffer_size <= 1:
        anomalies = np.array(list(zip(starts, ends - 1)))
        return y_true.astype(np.float64), anomalies

    y_true_cont = y_true.astype(np.float64)
    slope_length = buffer_size // 2
    length = y_true_cont.shape[0]
    for s, e in zip(starts, ends):
        e -= 1
        x1 = np.arange(e, min(e + slope_length, length))
        y_true_cont[x1] += np.sqrt(1 - (x1 - e) / buffer_size)
        x2 = np.arange(max(s - slope_length, 0), s)
        y_true_cont[x2] += np.sqrt(1 - (s - x2) / buffer_size)
    y_true_cont = np.clip(y_true_cont, 0, 1)
    starts, ends = _anomaly_bounds(y_true_cont)
    anomalies = np.array(list(zip(starts, ends - 1)))

    return y_true_cont, anomalies


def _uniform_threshold_sampling(y_score: np.ndarray) -> np.ndarray:
    """Create the threshold via uniform sampling."""
    # magic number from original implementation
    n_samples = 250
    thresholds = np.sort(y_score)[::-1]
    thresholds = thresholds[
        np.linspace(0, thresholds.shape[0] - 1, n_samples, dtype=np.int_)
    ]
    return thresholds


def range_pr_roc_auc_support(
    y_true: np.ndarray,
    y_score: np.ndarray,
    buffer_size: int | None = None,
    skip_check: bool = False,
) -> tuple[float, float]:
    """Compute the range-based PR and ROC AUC.

    Computes the area under the precision-recall-curve and the area under the
    receiver operating characteristic using the range-based precision and range-based
    recall definition from Paparrizos et al. published at VLDB 2022 [1]_.

    We first extend the anomaly labels by two slopes of ``buffer_size//2`` length on
    both sides of each anomaly, uniformly sample thresholds from the anomaly score, and
    then compute the confusion matrix for all thresholds. Using the resulting precision
    and recall values, we can plot a curve and compute its area.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    buffer_size : int, optional
        Size of the buffer region around an anomaly. We add an increasing slope of size
        ``buffer_size//2`` to the beginning of anomalies and a decreasing slope of size
        ``buffer_size//2`` to the end of anomalies. Per default
        (when ``buffer_size==None``), ``buffer_size`` is the median length of the
        anomalies within the time series. However, you can also set it to the period
        size of the dominant frequency or any other desired value.
    skip_check : bool, default False
        Whether to skip the input checks.

    Returns
    -------
    Tuple[float, float]
        Range-based PR AUC and range-based ROC AUC.

    References
    ----------
    .. [1] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay,
       Aaron Elmore, and Michael J. Franklin. Volume Under the Surface: A New Accuracy
       Evaluation Measure for Time-Series Anomaly Detection. PVLDB, 15(11):
       2774 - 2787, 2022.
       doi:`10.14778/3551793.3551830 <https://doi.org/10.14778/3551793.3551830>`_
    """
    if not skip_check:
        y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    y_true_cont, anomalies = _extend_anomaly_labels(y_true, buffer_size)
    thresholds = _uniform_threshold_sampling(y_score)
    p = np.average([np.sum(y_true), np.sum(y_true_cont)])

    recalls = np.zeros(thresholds.shape[0] + 2)  # tprs
    fprs = np.zeros(thresholds.shape[0] + 2)
    precisions = np.ones(thresholds.shape[0] + 1)

    for i, t in enumerate(thresholds):
        y_pred = y_score >= t
        product = y_true_cont * y_pred
        tp = np.sum(product)
        # fp = np.dot((np.ones_like(y_pred) - y_true_cont).T, y_pred)
        fp = np.sum(y_pred) - tp
        n = len(y_pred) - p

        existence_reward = [np.sum(product[s : e + 1]) > 0 for s, e in anomalies]
        existence_reward = np.sum(existence_reward) / anomalies.shape[0]

        recall = min(tp / p, 1) * existence_reward  # = tpr
        fpr = min(fp / n, 1)
        precision = tp / np.sum(y_pred)

        recalls[i + 1] = recall
        fprs[i + 1] = fpr
        precisions[i + 1] = precision

    recalls[-1] = 1
    fprs[-1] = 1

    range_pr_auc = np.sum(
        (recalls[1:-1] - recalls[:-2]) * (precisions[1:] + precisions[:-1]) / 2
    )
    range_roc_auc = np.sum((fprs[1:] - fprs[:-1]) * (recalls[1:] + recalls[:-1]) / 2)
    return range_pr_auc, range_roc_auc


def range_roc_auc_score(
    y_true: np.ndarray, y_score: np.ndarray, buffer_size: int | None = None
) -> float:
    """Compute the range-based area under the ROC curve.

    Computes the area under the receiver-operating-characteristic-curve using the
    range-based TPR and range-based FPR definition from Paparrizos et al.
    published at VLDB 2022 [1]_.

    We first extend the anomaly labels by two slopes of ``buffer_size//2`` length on
    both sides of each anomaly, uniformly sample thresholds from the anomaly score, and
    then compute the confusion matrix for all thresholds. Using the resulting precision
    and recall values, we can plot a curve and compute its area.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    buffer_size : int, optional
        Size of the buffer region around an anomaly. We add an increasing slope of size
        ``buffer_size//2`` to the beginning of anomalies and a decreasing slope of size
        ``buffer_size//2`` to the end of anomalies. Per default
        (when ``buffer_size==None``), ``buffer_size`` is the median length of the
        anomalies within the time series. However, you can also set it to the period
        size of the dominant frequency or any other desired value.

    Returns
    -------
    Tuple[float, float]
        Range-based ROC AUC score.

    References
    ----------
    .. [1] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay,
       Aaron Elmore, and Michael J. Franklin. Volume Under the Surface: A New Accuracy
       Evaluation Measure for Time-Series Anomaly Detection. PVLDB, 15(11):
       2774 - 2787, 2022.
       doi:`10.14778/3551793.3551830 <https://doi.org/10.14778/3551793.3551830>`_
    """
    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    _, range_auc_roc = range_pr_roc_auc_support(
        y_true, y_score, buffer_size, skip_check=True
    )
    return range_auc_roc


def range_pr_auc_score(
    y_true: np.ndarray, y_score: np.ndarray, buffer_size: int | None = None
) -> float:
    """Compute the area under the range-based PR curve.

    Computes the area under the precision-recall-curve using the range-based precision
    and range-based recall definition from Paparrizos et al. published at VLDB 2022
    [1]_.

    We first extend the anomaly labels by two slopes of ``buffer_size//2`` length on
    both sides of each anomaly, uniformly sample thresholds from the anomaly score, and
    then compute the confusion matrix for all thresholds. Using the resulting precision
    and recall values, we can plot a curve and compute its area.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    buffer_size : int, optional
        Size of the buffer region around an anomaly. We add an increasing slope of size
        ``buffer_size//2`` to the beginning of anomalies and a decreasing slope of size
        ``buffer_size//2`` to the end of anomalies. Per default
        (when ``buffer_size==None``), ``buffer_size`` is the median length of the
        anomalies within the time series. However, you can also set it to the period
        size of the dominant frequency or any other desired value.

    Returns
    -------
    Tuple[float, float]
        Range-based PR AUC score.

    References
    ----------
    .. [1] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay,
       Aaron Elmore, and Michael J. Franklin. Volume Under the Surface: A New Accuracy
       Evaluation Measure for Time-Series Anomaly Detection. PVLDB, 15(11):
       2774 - 2787, 2022.
       doi:`10.14778/3551793.3551830 <https://doi.org/10.14778/3551793.3551830>`_
    """
    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    range_pr_auc, _ = range_pr_roc_auc_support(
        y_true, y_score, buffer_size, skip_check=True
    )
    return range_pr_auc


def range_pr_vus_score(
    y_true: np.ndarray, y_score: np.ndarray, max_buffer_size: int = 500
) -> float:
    """Compute the range-based PR VUS score.

    Computes the volume under the precision-recall-buffer_size-surface using the
    range-based precision and range-based recall definition from Paparrizos et al.
    published at VLDB 2022 [1]_.

    For all buffer sizes from 0 to ``max_buffer_size``, we first extend the anomaly
    labels by two slopes of ``buffer_size//2`` length on both sides of each anomaly,
    uniformly sample thresholds from the anomaly score, and then compute the confusion
    matrix for all thresholds. Using the resulting precision and recall values, we can
    plot a curve and compute its area.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    max_buffer_size : int, default=500
        Maximum size of the buffer region around an anomaly. We iterate over all buffer
        sizes from 0 to ``may_buffer_size`` to create the surface.

    Returns
    -------
    Tuple[float, float]
        Range-based PR VUS score.

    References
    ----------
    .. [1] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay,
       Aaron Elmore, and Michael J. Franklin. Volume Under the Surface: A New Accuracy
       Evaluation Measure for Time-Series Anomaly Detection. PVLDB, 15(11):
       2774 - 2787, 2022.
       doi:`10.14778/3551793.3551830 <https://doi.org/10.14778/3551793.3551830>`_
    """
    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    prs = np.zeros(max_buffer_size + 1)
    for buffer_size in np.arange(0, max_buffer_size + 1):
        pr_auc, _ = range_pr_roc_auc_support(
            y_true, y_score, buffer_size=buffer_size, skip_check=True
        )
        prs[buffer_size] = pr_auc
    range_pr_volume = np.sum(prs) / (max_buffer_size + 1)
    return range_pr_volume


def range_roc_vus_score(
    y_true: np.ndarray, y_score: np.ndarray, max_buffer_size: int = 500
) -> float:
    """Compute the range-based ROC VUS score.

    Computes the volume under the receiver-operating-characteristic-buffer_size-surface
    using the range-based TPR and range-based FPR definition from Paparrizos et al.
    published at VLDB 2022 [1]_.

    For all buffer sizes from 0 to ``max_buffer_size``, we first extend the anomaly
    labels by two slopes of ``buffer_size//2`` length on both sides of each anomaly,
    uniformly sample thresholds from the anomaly score, and then compute the confusion
    matrix for all thresholds. Using the resulting false positive (FPR) and false
    positive rates (FPR), we can plot a curve and compute its area.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    max_buffer_size : int, default=500
        Maximum size of the buffer region around an anomaly. We iterate over all buffer
        sizes from 0 to ``may_buffer_size`` to create the surface.

    Returns
    -------
    Tuple[float, float]
        Range-based PR VUS score.

    References
    ----------
    .. [1] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay,
       Aaron Elmore, and Michael J. Franklin. Volume Under the Surface: A New Accuracy
       Evaluation Measure for Time-Series Anomaly Detection. PVLDB, 15(11):
       2774 - 2787, 2022.
       doi:`10.14778/3551793.3551830 <https://doi.org/10.14778/3551793.3551830>`_
    """
    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    rocs = np.zeros(max_buffer_size + 1)
    for buffer_size in np.arange(0, max_buffer_size + 1):
        _, roc_auc = range_pr_roc_auc_support(
            y_true, y_score, buffer_size=buffer_size, skip_check=True
        )
        rocs[buffer_size] = roc_auc
    range_roc_volume = np.sum(rocs) / (max_buffer_size + 1)
    return range_roc_volume
