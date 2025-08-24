"""Metrics on binary predictions for anomaly detection."""

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["range_precision", "range_recall", "range_f_score"]

import warnings

import numpy as np

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

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_pred : np.ndarray
        Anomaly scores for each point of the time series of shape (n_instances,).
    alpha : float
        Default is 0 = no existence reward.
        Because precision emphasizes prediction quality, there is no need for an
        existence reward and this value should always be set to 0 [1].
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
        y_pred_ranges, y_true_ranges, gamma=cardinality, bias_type=bias, alpha=alpha
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
        y_pred_ranges, y_true_ranges, gamma=cardinality, bias_type=bias, alpha=alpha
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
        Default is 0 = no existence reward for precision.
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
    y_true, y_pred = check_y(y_true, y_pred, force_y_pred_continuous=False)
    if np.unique(y_pred).shape[0] == 1:
        warnings.warn(
            "Cannot compute metric for a constant value in y_score, returning 0.0!",
            stacklevel=2,
        )
        return 0.0

    y_pred_ranges = _binary_to_ranges(y_pred)
    y_true_ranges = _binary_to_ranges(y_true)

    precision = _ts_precision(
        y_pred_ranges, y_true_ranges, gamma=cardinality, bias_type=p_bias, alpha=p_alpha
    )
    recall = _ts_recall(
        y_pred_ranges, y_true_ranges, gamma=cardinality, bias_type=r_bias, alpha=r_alpha
    )

    if precision + recall > 0:
        fscore = ((1 + beta**2) * (precision * recall)) / (
            beta**2 * (precision + recall)
        )
    else:
        fscore = 0.0

    return fscore


def _flatten_ranges(ranges):
    """
    If the input is a list of lists, it flattens it into a single list.

    Parameters
    ----------
    ranges : list of tuples or list of lists of tuples
        The ranges to flatten. each tuple shoulod be in the format of (start, end).

    Returns
    -------
    list of tuples
        A flattened list of ranges.

    Examples
    --------
    >>> _flatten_ranges([[(1, 5), (10, 15)], [(20, 25)]])
    [(1, 5), (10, 15), (20, 25)]
    """
    if not ranges:
        return []
    if isinstance(ranges[0], list):
        flat = []
        for sublist in ranges:
            for pred in sublist:
                flat.append(pred)
        return flat
    return ranges


def _udf_gamma_def(overlap_count):
    """User-defined gamma function. Should return a gamma value > 1.

    Parameters
    ----------
    overlap_count : int
        The number of overlapping ranges.

    Returns
    -------
    float
        The user-defined gamma value (>1).
    """
    return_val = 1 + 0.1 * overlap_count  # modify this function as needed

    return return_val


def _calculate_bias(position, length, bias_type="flat"):
    """Calculate bias value based on position and length.

    Parameters
    ----------
    position : int
        Current position in the range
    length : int
        Total length of the range
    bias_type : str, default="flat"
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].
    """
    if bias_type == "flat":
        return 1.0
    elif bias_type == "front":
        return 1.0 - (position - 1) / length
    elif bias_type == "middle":
        if length / 2 == 0:
            return 1.0
        if position <= length / 2:
            return position / (length / 2)
        else:
            return (length - position + 1) / (length / 2)
    elif bias_type == "back":
        return position / length
    else:
        raise ValueError(f"Invalid bias type: {bias_type}")


def _gamma_select(cardinality, gamma):
    """Select a gamma value based on the cardinality type.

    Parameters
    ----------
    cardinality : int
        The number of overlapping ranges.
    gamma : str
        Gamma to use. Should be one of ["one", "reciprocal", "udf_gamma"].

    Returns
    -------
    float
        The selected gamma value.

    Raises
    ------
    ValueError
        If an invalid `gamma` type is provided or if `udf_gamma` is required
        but not provided.
    """
    if gamma == "one":
        return 1.0
    elif gamma == "reciprocal":
        return 1 / cardinality if cardinality > 1 else 1.0
    elif gamma == "udf_gamma":
        if _udf_gamma_def(cardinality) is not None:
            return 1.0 / _udf_gamma_def(cardinality)
        else:
            raise ValueError("udf_gamma must be provided for 'udf_gamma' gamma type.")
    else:
        raise ValueError(
            "Invalid gamma type. Choose from ['one', 'reciprocal', 'udf_gamma']."
        )


def _calculate_overlap_reward_precision(pred_range, overlap_set, bias_type):
    """Overlap Reward for y_pred.

    Parameters
    ----------
    pred_range : tuple
        The predicted range.
    overlap_set : set
        The set of overlapping positions.
    bias_type : str
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].

    Returns
    -------
    float
        The weighted value for overlapping positions only.
    """
    start, end = pred_range
    length = end - start + 1

    max_value = 0  # Total possible weighted value for all positions.
    my_value = 0  # Weighted value for overlapping positions only.

    for i in range(1, length + 1):
        global_position = start + i - 1
        bias_value = _calculate_bias(i, length, bias_type)
        max_value += bias_value

        if global_position in overlap_set:
            my_value += bias_value

    return my_value / max_value if max_value > 0 else 0.0


def _calculate_overlap_reward_recall(real_range, overlap_set, bias_type):
    """Overlap Reward for y_real.

    Parameters
    ----------
    real_range : tuple
        The real range.
    overlap_set : set
        The set of overlapping positions.
    bias_type : str
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].

    Returns
    -------
    float
        The weighted value for overlapping positions only.
    """
    start, end = real_range
    length = end - start + 1

    max_value = 0.0  # Total possible weighted value for all positions.
    my_value = 0.0  # Weighted value for overlapping positions only.

    for i in range(1, length + 1):
        global_position = start + i - 1
        bias_value = _calculate_bias(i, length, bias_type)
        max_value += bias_value

        if global_position in overlap_set:
            my_value += bias_value

    return my_value / max_value if max_value > 0 else 0.0


def _binary_to_ranges(binary_sequence):
    """
    Convert a binary sequence to a list of anomaly ranges.

    Parameters
    ----------
    binary_sequence : list
        Binary sequence where 1 indicates anomaly and 0 indicates normal.

    Returns
    -------
    list of tuples
        List of anomaly ranges as (start, end) tuples.

    """
    ranges = []
    start = None

    for i, val in enumerate(binary_sequence):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ranges.append((start, i - 1))
            start = None

    if start is not None:
        ranges.append((start, len(binary_sequence) - 1))

    return ranges


def _ts_precision(
    y_pred_ranges, y_real_ranges, gamma="one", bias_type="flat", alpha=0.0
):
    """
    Implement range-based precision for time series anomaly detection.

    Parameters
    ----------
    y_pred : list of predicted anomaly ranges: each tuple in the list represents an
        interval [start, end) of a detected anomaly.
    y_real : list of true anomaly ranges: each tuple in the list represents an interval
        [start, end) of a true anomaly.
    bias_type : str, default="flat"
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
    gamma : str, default="one"
        Cardinality type. Should be one of ["reciprocal", "one"].
    alpha : float
        Default is 0 = no existence reward.
        Because precision emphasizes prediction quality, there is no need for an
        existence reward and this value should always be set to 0.

    Returns
    -------
    float
        Range-based precision
    """
    if gamma not in ["reciprocal", "one"]:
        raise ValueError("Invalid gamma type for precision. Use 'reciprocal' or 'one'.")

    if bias_type not in [
        "flat",
        "front",
        "middle",
        "back",
    ]:
        raise ValueError(
            "Invalid bias type. Choose from ['flat', 'front', 'middle', 'back']."
        )

    total_overlap_reward = 0.0
    total_cardinality = 0

    for pred_range in y_pred_ranges:
        overlap_set = set()
        cardinality = 0

        for real_start, real_end in y_real_ranges:
            overlap_start = max(pred_range[0], real_start)
            overlap_end = min(pred_range[1], real_end)

            if overlap_start <= overlap_end:
                overlap_set.update(range(overlap_start, overlap_end + 1))
                cardinality += 1

        overlap_reward = _calculate_overlap_reward_precision(
            pred_range, overlap_set, bias_type
        )

        existence_reward = 1.0 if overlap_set else 0.0
        gamma_value = _gamma_select(cardinality, gamma)

        score_i = alpha * existence_reward + (1 - alpha) * (
            gamma_value * overlap_reward
        )

        total_overlap_reward += score_i

        total_cardinality += 1

    precision = total_overlap_reward / total_cardinality if total_cardinality else 0.0

    return precision


def _ts_recall(y_pred_ranges, y_real_ranges, gamma="one", bias_type="flat", alpha=0.0):
    """
    Implement range-based recall for time series anomaly detection.

    Parameters
    ----------
    y_pred : list of predicted anomaly ranges: each tuple in the list represents an
        interval [start, end) of a detected anomaly.
    y_real : list of true anomaly ranges: each tuple in the list represents an interval
        [start, end) of a true anomaly.
    gamma : str, default="one"
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
    bias_type : str, default="flat"
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
    alpha : float, default: 0.0
        Weight for existence reward in recall calculation.

    Returns
    -------
    float
        Range-based recall
    """
    if gamma not in ["reciprocal", "one", "udf_gamma"]:
        raise ValueError("Invalid gamma type for precision. Use 'reciprocal' or 'one'.")

    if bias_type not in [
        "flat",
        "front",
        "middle",
        "back",
    ]:
        raise ValueError(
            "Invalid bias type. Choose from ['flat', 'front', 'middle', 'back']."
        )

    total_overlap_reward = 0.0

    for real_range in y_real_ranges:
        overlap_set = set()
        cardinality = 0

        for pred_range in y_pred_ranges:
            overlap_start = max(real_range[0], pred_range[0])
            overlap_end = min(real_range[1], pred_range[1])

            if overlap_start <= overlap_end:
                overlap_set.update(range(overlap_start, overlap_end + 1))
                cardinality += 1

        existence_reward = 1.0 if overlap_set else 0.0

        if overlap_set:
            overlap_reward = _calculate_overlap_reward_recall(
                real_range, overlap_set, bias_type
            )
            gamma_value = _gamma_select(cardinality, gamma)
            overlap_reward *= gamma_value
        else:
            overlap_reward = 0.0

        recall_score = alpha * existence_reward + (1 - alpha) * overlap_reward
        total_overlap_reward += recall_score

    recall = total_overlap_reward / len(y_real_ranges) if y_real_ranges else 0.0
    return recall
