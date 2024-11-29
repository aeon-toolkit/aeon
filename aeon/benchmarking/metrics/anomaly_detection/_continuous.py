"""Metrics on anomaly scores for anomaly detection."""

from __future__ import annotations

__maintainer__ = ["SebastianSchmidl"]
__all__ = [
    "roc_auc_score",
    "pr_auc_score",
    "f_score_at_k_points",
    "f_score_at_k_ranges",
    "rp_rr_auc_score",
]


import warnings

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve
from sklearn.metrics import roc_auc_score as _roc_auc_score

from aeon.benchmarking.metrics.anomaly_detection._util import check_y
from aeon.benchmarking.metrics.anomaly_detection.thresholding import (
    top_k_points_threshold,
    top_k_ranges_threshold,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the ROC AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).

    Returns
    -------
    float
        ROC AUC score.

    See Also
    --------
    sklearn.metrics.roc_auc_score
        Is used internally.
    """
    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0
    return _roc_auc_score(y_true, y_score)


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the precision-recall AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).

    Returns
    -------
    float
        Precision-recall AUC score.

    See Also
    --------
    sklearn.metrics.precision_recall_curve
        Function used under the hood.
    """
    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    x, y, _ = precision_recall_curve(y_true, np.array(y_score))
    area = auc(y, x)
    return area


def f_score_at_k_points(
    y_true: np.ndarray, y_score: np.ndarray, k: int | None = None
) -> float:
    """Compute the F-score at k based on single points.

    This metric only considers the top-k predicted anomalous points within the scoring
    by finding a threshold on the scoring that produces at least k anomalous points. If
    `k` is not specified, the number of anomalies within the ground truth is used as
    `k`.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    k : int (optional)
        Number of top anomalies used to calculate precision. If `k` is not specified
        (`None`) the number of true anomalies (based on the ground truth values) is
        used.

    Returns
    -------
    float
        F1 score at k.

    See Also
    --------
    aeon.benchmarking.metrics.anomaly_detection.thresholding.top_k_points_threshold
        Function used to find the threshold.
    """
    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    threshold = top_k_points_threshold(y_true, y_score, k)
    y_pred = y_score >= threshold
    return f1_score(y_true, y_pred)


def f_score_at_k_ranges(
    y_true: np.ndarray, y_score: np.ndarray, k: int | None = None
) -> float:
    """Compute the range-based F-score at k based on anomaly ranges.

    This metric only considers the top-k predicted anomaly ranges within the scoring by
    finding a threshold on the scoring that produces at least k anomalous ranges. If `k`
    is not specified, the number of anomalies within the ground truth is used as `k`.

    This implementation uses the community package
    `prts <https://pypi.org/project/prts/>`_ as a soft-dependency to compute the
    range-based F-score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    k : int (optional)
        Number of top anomalies used to calculate precision. If `k` is not specified
        (`None`) the number of true anomalies (based on the ground truth values) is
        used.

    Returns
    -------
    float
        F1 score at k.

    See Also
    --------
    aeon.benchmarking.metrics.anomaly_detection.thresholding.top_k_ranges_threshold
        Function used to find the threshold.
    """
    _check_soft_dependencies(
        "prts", obj="f_score_at_k_ranges", suppress_import_stdout=True
    )

    from prts import ts_fscore

    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    threshold = top_k_ranges_threshold(y_true, y_score, k)
    y_pred = y_score >= threshold
    return ts_fscore(y_true, y_pred, p_alpha=1, r_alpha=1, cardinality="reciprocal")


def rp_rr_auc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_samples: int = 50,
    r_alpha: float = 0.5,
    p_alpha: float = 0,
    cardinality: str = "reciprocal",
    bias: str = "flat",
) -> float:
    """Compute the AUC-score of the range-based precision-recall curve.

    Computes the area under the precision recall curve when using the range-based
    precision and range-based recall metric introduced by Tatbul et al. at NeurIPS 2018
    [1]_. This implementation uses the community package
    `prts <https://pypi.org/project/prts/>`_ as a soft-dependency.

    This metric only considers the top-k predicted anomaly ranges within the scoring by
    finding a threshold on the scoring that produces at least k anomalous ranges. If `k`
    is not specified, the number of anomalies within the ground truth is used as `k`.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_score : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    max_samples: int
        The implementation of the range-based precision and recall metrics is quite
        slow because it relies on the non-optimized ``prts``-package. To prevent long
        runtimes caused by scorings with high precision (many thresholds), just a
        specific amount of possible thresholds is sampled. This parameter controls the
        maximum number of thresholds; however, too low numbers degrade the metrics'
        quality.
    r_alpha : float
        Weight of the existence reward for the range-based recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when
        not all - cases, `p_alpha` should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.

    Returns
    -------
    float
        Area under the range-based precision-recall curve.

    References
    ----------
    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin
       Gottschlich. "Precision and Recall for Time Series." In Proceedings of the
       International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018.
       http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """
    _check_soft_dependencies(
        "prts", obj="f_score_at_k_ranges", suppress_import_stdout=True
    )

    from prts import ts_precision, ts_recall

    y_true, y_pred = check_y(y_true, y_score, force_y_pred_continuous=True)
    if np.unique(y_score).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    thresholds = np.unique(y_score)
    thresholds.sort()
    # The first precision and recall values are precision=class balance and recall=1.0,
    # which corresponds to a classifier that always predicts the positive class,
    # independently of the threshold. This means that we can skip the first threshold!
    p0 = y_true.sum() / len(y_true)
    r0 = 1.0
    thresholds = thresholds[1:]

    # sample thresholds
    n_thresholds = thresholds.shape[0]
    if n_thresholds > max_samples:
        every_nth = n_thresholds // (max_samples - 1)
        sampled_thresholds = thresholds[::every_nth]
        if thresholds[-1] == sampled_thresholds[-1]:
            thresholds = sampled_thresholds
        else:
            thresholds = np.r_[sampled_thresholds, thresholds[-1]]

    recalls = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        y_pred = (y_score >= threshold).astype(np.int64)
        recalls[i] = ts_recall(
            y_true, y_pred, alpha=r_alpha, cardinality=cardinality, bias=bias
        )
        precisions[i] = ts_precision(
            y_true, y_pred, alpha=p_alpha, cardinality=cardinality, bias=bias
        )
    # first sort by recall, then by precision to break ties
    # (important for noisy scorings)
    sorted_idx = np.lexsort((precisions * (-1), recalls))[::-1]

    # add first and last points to the curve
    recalls = np.r_[r0, recalls[sorted_idx], 0]
    precisions = np.r_[p0, precisions[sorted_idx], 1]

    # calculate area under the curve
    return auc(recalls, precisions)
