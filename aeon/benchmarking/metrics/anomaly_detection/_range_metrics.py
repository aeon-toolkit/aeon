"""Metrics on binary predictions for anomaly detection."""

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["range_precision", "range_recall", "range_f_score"]

import warnings

import numpy as np

from aeon.benchmarking.metrics.anomaly_detection._range_ts_metrics import (
    _binary_to_ranges,
    _ts_precision,
    _ts_recall,
)
from aeon.benchmarking.metrics.anomaly_detection._util import check_y


def range_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0,
    cardinality: str = "reciprocal",
    bias: str = "flat",
) -> float:
    """Compute the range-based precision metric.

    Range-based metrics were introduced by Tatbul et al. at NeurIPS 2018 [1]_.

    Range precision is the average precision of each predicted anomaly range. For each
    predicted continuous anomaly range the overlap size, position, and cardinality is
    considered. For more details, please refer to the paper [1]_.

    The `alpha` parameter for the existence reward was removed. Because precision
    emphasizes prediction quality, there is no need for an existence reward and this
    value should always be set to 0.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_pred : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    alpha : float
        DEPRECATED. Default is 0 = no existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.

    Returns
    -------
    float
        Range-based precision

    References
    ----------
    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin
       Gottschlich. "Precision and Recall for Time Series." In Proceedings of the
       International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018.
       http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """
    if alpha != 0:
        warnings.warn(
            "The alpha parameter should not be used in range precision. This "
            "parameter is removed in 1.3.0.",
            stacklevel=2,
            category=FutureWarning,
        )
    y_true, y_pred = check_y(y_true, y_pred, force_y_pred_continuous=False)
    if np.unique(y_pred).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    y_pred_ranges = _binary_to_ranges(y_pred)
    y_true_ranges = _binary_to_ranges(y_true)
    return _ts_precision(
        y_pred_ranges, y_true_ranges, gamma=cardinality, bias_type=bias
    )


def range_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0,
    cardinality: str = "reciprocal",
    bias: str = "flat",
) -> float:
    """Compute the range-based recall metric.

    Range-based metrics were introduced by Tatbul et al. at NeurIPS 2018 [1]_.

    Range recall is the average recall of each real anomaly range. For each real
    anomaly range the overlap size, position, and cardinality with predicted anomaly
    ranges are considered. In addition, an existence reward can be given that boosts
    the recall even if just a single point of the real anomaly is in the predicted
    ranges. For more details, please refer to the paper [1]_.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_pred : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only existence
        reward. The existence reward is given if the real anomaly range has overlap
        with even a single point of the predicted anomaly range.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.

    Returns
    -------
    float
        Range-based recall

    References
    ----------
    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin
       Gottschlich. "Precision and Recall for Time Series." In Proceedings of the
       International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018.
       http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """
    y_true, y_pred = check_y(y_true, y_pred, force_y_pred_continuous=False)
    if np.unique(y_pred).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    y_pred_ranges = _binary_to_ranges(y_pred)
    y_true_ranges = _binary_to_ranges(y_true)
    return _ts_recall(
        y_pred_ranges, y_true_ranges, alpha=alpha, gamma=cardinality, bias_type=bias
    )


def range_f_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1,
    p_alpha: float = 0,
    r_alpha: float = 0.5,
    cardinality: str = "reciprocal",
    p_bias: str = "flat",
    r_bias: str = "flat",
) -> float:
    """Compute the F-score using the range-based recall and precision metrics.

    Range-based metrics were introduced by Tatbul et al. at NeurIPS 2018 [1]_.

    The F-beta score is the weighted harmonic mean of precision and recall, reaching
    its optimal value at 1 and its worst value at 0. This implementation uses the
    range-based precision and range-based recall as basis.

    The `p_alpha` parameter for the potential existance reward in the calculation of
    range-based precision was removed. `p_alpha` should always be set to 0, anyway.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_pred : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    beta : float
        F-score beta determines the weight of recall in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
    p_alpha : float
        DEPRECATED. Default is 0 = no existence reward for precision.
    r_alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only
        existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    p_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    r_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.

    Returns
    -------
    float
        Range-based F-score

    References
    ----------
    .. [1] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin
       Gottschlich. "Precision and Recall for Time Series." In Proceedings of the
       International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018.
       http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """
    if p_alpha != 0:
        warnings.warn(
            "The p_alpha parameter should not be used. This parameter is removed "
            "in 1.3.0.",
            stacklevel=2,
            category=FutureWarning,
        )

    y_true, y_pred = check_y(y_true, y_pred, force_y_pred_continuous=False)
    if np.unique(y_pred).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    y_pred_ranges = _binary_to_ranges(y_pred)
    y_true_ranges = _binary_to_ranges(y_true)

    precision = _ts_precision(y_pred_ranges, y_true_ranges, cardinality, p_bias)
    recall = _ts_recall(y_pred_ranges, y_true_ranges, cardinality, r_bias, r_alpha)

    if precision + recall > 0:
        fscore = ((1 + beta**2) * (precision * recall)) / (
            beta**2 * (precision + recall)
        )
    else:
        fscore = 0.0

    return fscore
