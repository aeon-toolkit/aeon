"""Calculate Precision, Recall, and F1-Score for time series anomaly detection."""

maintainer = []
__all__ = ["ts_precision", "ts_recall", "ts_fscore"]


def calculate_bias(position, length, bias_type="flat"):
    """Calculate bias value based on position and length.

    Parameters
    ----------
    position : int
        Current position in the range
    length : int
        Total length of the range
    bias_type : str
        Type of bias to apply, Should be one of ["flat", "front", "middle", "back"].
        (default: "flat")
    """
    if bias_type == "flat":
        return 1.0
    elif bias_type == "front":
        return 1.0 - (position - 1) / length
    elif bias_type == "middle":
        return 1.0 - abs(2 * (position - 1) / (length - 1) - 1) if length > 1 else 1.0
    elif bias_type == "back":
        return position / length
    else:
        raise ValueError(f"Invalid bias type: {bias_type}")


def gamma_select(cardinality, gamma, udf_gamma=None):
    """Select a gamma value based on the cardinality type."""
    if gamma == "one":
        return 1.0
    elif gamma == "reciprocal":
        return 1 / cardinality if cardinality > 1 else 1.0
    elif gamma == "udf_gamma":
        if udf_gamma is not None:
            return 1.0 / udf_gamma
        else:
            raise ValueError("udf_gamma must be provided for 'udf_gamma' gamma type.")
    else:
        raise ValueError("Invalid gamma type.")


def calculate_overlap_reward_precision(pred_range, overlap_set, bias_type):
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
        bias_value = calculate_bias(i, length, bias_type)
        max_value += bias_value

        if global_position in overlap_set:
            my_value += bias_value

    return my_value / max_value if max_value > 0 else 0.0


def calculate_overlap_reward_recall(real_range, overlap_set, bias_type):
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
        bias_value = calculate_bias(i, length, bias_type)
        max_value += bias_value

        if global_position in overlap_set:
            my_value += bias_value

    return my_value / max_value if max_value > 0 else 0.0


def ts_precision(y_pred, y_real, gamma="one", bias_type="flat", udf_gamma=None):
    """Precision for either a single set or the entire time series.

    Parameters
    ----------
    y_pred : list of tuples or list of list of tuples
        The predicted ranges.
    y_real : list of tuples
        The real ranges.
    gamma : str
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
        (default: "one")
    bias_type : str
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
        (default: "flat")
    udf_gamma : int or None
        User-defined gamma value. (default: None)

    Returns
    -------
    float
        Range-based precision
    """
    """
    example:
    y_pred = [(1, 3), (5, 7)]
    y_real = [(2, 6), (8, 10)]
    """
    #  Check if the input is a single set of predicted ranges or multiple sets
    if isinstance(y_pred[0], tuple):
        # y_pred is a single set of predicted ranges
        total_overlap_reward = 0.0
        total_cardinality = 0

        for pred_range in y_pred:
            overlap_set = set()
            cardinality = 0

            for real_start, real_end in y_real:
                overlap_start = max(pred_range[0], real_start)
                overlap_end = min(pred_range[1], real_end)

                if overlap_start <= overlap_end:
                    overlap_set.update(range(overlap_start, overlap_end + 1))
                    cardinality += 1

            overlap_reward = calculate_overlap_reward_precision(
                pred_range, overlap_set, bias_type
            )
            gamma_value = gamma_select(cardinality, gamma, udf_gamma)

            total_overlap_reward += gamma_value * overlap_reward
            total_cardinality += 1

        return (
            total_overlap_reward / total_cardinality if total_cardinality > 0 else 0.0
        )

    else:
        """
        example:
        y_pred = [[(1, 3), (5, 7)],[(10, 12)]]
        y_real = [(2, 6), (8, 10)]
        """
        # y_pred as multiple sets of predicted ranges
        total_precision = 0.0
        total_ranges = 0

        for pred_ranges in y_pred:  # Iterate over all sets of predicted ranges
            precision = ts_precision(
                pred_ranges, y_real, gamma, bias_type, udf_gamma
            )  # Recursive call for single sets
            total_precision += precision * len(pred_ranges)
            total_ranges += len(pred_ranges)

        return total_precision / total_ranges if total_ranges > 0 else 0.0


def ts_recall(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None):
    """Calculate Recall for time series anomaly detection.

    Parameters
    ----------
    y_pred : list of tuples or list of list of tuples
        The predicted ranges.
    y_real : list of tuples or list of list of tuples
        The real ranges.
    gamma : str
        Cardinality type. Should be one of ["reciprocal", "one", "udf_gamma"].
        (default: "one")
    bias_type : str
        Type of bias to apply. Should be one of ["flat", "front", "middle", "back"].
        (default: "flat")
    alpha : float
        Weight for existence reward in recall calculation. (default: 0.0)
    udf_gamma : int or None
        User-defined gamma value. (default: None)

    Returns
    -------
    float
        Range-based recall
    """
    if isinstance(y_real[0], tuple):  # Single set of real ranges
        total_overlap_reward = 0.0

        for real_range in y_real:
            overlap_set = set()
            cardinality = 0

            for pred_range in y_pred:
                overlap_start = max(real_range[0], pred_range[0])
                overlap_end = min(real_range[1], pred_range[1])

                if overlap_start <= overlap_end:
                    overlap_set.update(range(overlap_start, overlap_end + 1))
                    cardinality += 1

            # Existence Reward
            existence_reward = 1.0 if overlap_set else 0.0

            if overlap_set:
                overlap_reward = calculate_overlap_reward_recall(
                    real_range, overlap_set, bias_type
                )
                gamma_value = gamma_select(cardinality, gamma, udf_gamma)
                overlap_reward *= gamma_value
            else:
                overlap_reward = 0.0

            # Total Recall Score
            recall_score = alpha * existence_reward + (1 - alpha) * overlap_reward
            total_overlap_reward += recall_score

        return total_overlap_reward / len(y_real) if y_real else 0.0

    elif isinstance(y_real[0], list):  # Multiple sets of real ranges
        total_recall = 0.0
        total_real = 0

        for real_ranges in y_real:  # Iterate over all sets of real ranges
            recall = ts_recall(y_pred, real_ranges, gamma, bias_type, alpha, udf_gamma)
            total_recall += recall * len(real_ranges)
            total_real += len(real_ranges)

        return total_recall / total_real if total_real > 0 else 0.0


def ts_fscore(y_pred, y_real, gamma="one", bias_type="flat", alpha=0.0, udf_gamma=None):
    """Calculate F1-Score for time series anomaly detection."""
    precision = ts_precision(y_pred, y_real, gamma, bias_type, udf_gamma)
    recall = ts_recall(y_pred, y_real, gamma, bias_type, alpha, udf_gamma)

    if precision + recall > 0:
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        fscore = 0.0

    return fscore
