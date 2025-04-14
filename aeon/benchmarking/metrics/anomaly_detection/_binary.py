"""Metrics on binary predictions for anomaly detection."""

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["range_precision", "range_recall", "range_f_score"]

import warnings

import numpy as np
from deprecated.sphinx import deprecated

from aeon.benchmarking.metrics.anomaly_detection._util import check_y
from aeon.utils.validation._dependencies import _check_soft_dependencies


# TODO: Remove in v1.2.0
@deprecated(
    version="1.1.0",
    reason="range_precision is deprecated and will be removed in v1.2.0. "
    "Please use ts_precision from the range_metrics module instead.",
    category=FutureWarning,
)
def range_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0,
    cardinality: str = "reciprocal",
    bias: str = "flat",
) -> float:
    """Compute the range-based precision metric.

    Range-based metrics were introduced by Tatbul et al. at NeurIPS 2018 [1]_. This
    implementation uses the community package `prts <https://pypi.org/project/prts/>`_
    as a soft-dependency.

    Range precision is the average precision of each predicted anomaly range. For each
    predicted continuous anomaly range the overlap size, position, and cardinality is
    considered. For more details, please refer to the paper [1]_.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_pred : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    alpha : float
        Weight of the existence reward. Because precision by definition emphasizes on
        prediction quality, there is no need for an existence reward and this value
        should always be set to 0.
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
    _check_soft_dependencies("prts", obj="range_precision", suppress_import_stdout=True)

    from prts import ts_precision

    y_true, y_pred = check_y(y_true, y_pred, force_y_pred_continuous=False)
    if np.unique(y_pred).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0
    return ts_precision(y_true, y_pred, alpha=alpha, cardinality=cardinality, bias=bias)


# TODO: Remove in v1.2.0
@deprecated(
    version="1.1.0",
    reason="range_recall is deprecated and will be removed in v1.2.0. "
    "Please use ts_recall from the range_metrics module instead.",
    category=FutureWarning,
)
def range_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0,
    cardinality: str = "reciprocal",
    bias: str = "flat",
) -> float:
    """Compute the range-based recall metric.

    Range-based metrics were introduced by Tatbul et al. at NeurIPS 2018 [1]_. This
    implementation uses the community package `prts <https://pypi.org/project/prts/>`_
    as a soft-dependency.

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
    _check_soft_dependencies("prts", obj="range_recall", suppress_import_stdout=True)

    from prts import ts_recall

    y_true, y_pred = check_y(y_true, y_pred, force_y_pred_continuous=False)
    if np.unique(y_pred).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0
    return ts_recall(y_true, y_pred, alpha=alpha, cardinality=cardinality, bias=bias)


# TODO: Remove in v1.2.0
@deprecated(
    version="1.1.0",
    reason="range_f_score is deprecated and will be removed in v1.2.0. "
    "Please use ts_fscore from the range_metrics module instead.",
    category=FutureWarning,
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

    Range-based metrics were introduced by Tatbul et al. at NeurIPS 2018 [1]_. This
    implementation uses the community package `prts <https://pypi.org/project/prts/>`_
    as a soft-dependency.

    The F-beta score is the weighted harmonic mean of precision and recall, reaching
    its optimal value at 1 and its worst value at 0. This implementation uses the
    range-based precision and range-based recall as basis.

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
        Weight of the existence reward for the range-based precision. For most - when
        not all - cases, `p_alpha` should be set to 0.
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
    _check_soft_dependencies("prts", obj="range_recall", suppress_import_stdout=True)

    from prts import ts_fscore

    y_true, y_pred = check_y(y_true, y_pred, force_y_pred_continuous=False)
    if np.unique(y_pred).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0
    return ts_fscore(
        y_true,
        y_pred,
        beta=beta,
        p_alpha=p_alpha,
        r_alpha=r_alpha,
        cardinality=cardinality,
        p_bias=p_bias,
        r_bias=r_bias,
    )
