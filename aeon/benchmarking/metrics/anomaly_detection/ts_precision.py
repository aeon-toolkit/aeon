"""Calculate the precision of a time series anomaly detection model."""


class RangePrecision:
    """
    Calculates Precision for time series.

    Parameters
    ----------
    y_real : np.ndarray
        set of ground truth anomaly ranges (actual anomalies).

    y_pred : np.ndarray
        set of predicted anomaly ranges.

    cardinality : str, default="one"
        Number of overlaps between y_pred and y_real.

    gamma : float, default=1.0
        Overlpa Cardinality Factor. Penalizes or adjusts the metric based on
        the cardinality.
        Should be one of {'reciprocal', 'one', 'udf_gamma'}.

    alpha : float
        Weight of the existence reward. Because precision by definition emphasizes on
        prediction quality, there is no need for an existence reward and this value
        should always be set to 0.

    bias : str, default="flat"
        Captures importance of positional factors within anomaly ranges.
        Determines the weight given to specific portions of anomaly range
        when calculating overlap rewards.
        Should be one of {'flat', 'front', 'middle', 'back'}.

        'flat' - All positions are equally important.
        'front' - Front positions are more important.
        'middle' - Middle positions are more important.
        'back' - Back positions are more important.

    omega : float
        Measure the extent and overlap between y_pred and y_real.
        Considers the size and position of overlap and rewards. Should
        be a float value between 0 and 1.
    """

    def __init__(self, bias="flat", alpha=0.0, gamma=None):
        assert gamma in ["reciprocal", "one", "udf_gamma"], "Invalid gamma type"
        assert bias in ["flat", "front", "middle", "back"], "Invalid bias type"

        self.bias = bias
        self.alpha = alpha
        self.gamma = gamma

    def calculate_overlap_set(y_pred, y_real):
        """
        Calculate the overlap set for all predicted and real ranges.

        Parameters
        ----------
        y_pred : np.ndarray

        y_real : np.ndarray

        Returns
        -------
        list of sets : List where each set represents the 'overlap positions'
        for a predicted range.

        Example
        -------
        y_pred = [(1, 5), (10, 15)]
        y_real = [(3, 8), (4, 6), (12, 18)]

        Output -> [ {3, 4, 5}, {12, 13, 14, 15} ]
        """
        overlap_sets = []

        for pred_start, pred_end in y_pred:
            overlap_set = set()
            for real_start, real_end in y_real:
                overlap_start = max(pred_start, real_start)
                overlap_end = min(pred_end, real_end)

                if overlap_start <= overlap_end:
                    overlap_set.update(
                        range(overlap_start, overlap_end + 1)
                    )  # Update set with overlap positions
            overlap_sets.append(overlap_set)

        return overlap_sets

    def calculate_bias(i, length, bias_type="flat"):
        """Calculate the bias value for a given postion in a range.

        Parameters
        ----------
        i : int
            Position index within the range.

        length : int
            Total length of the range.
        """
        if bias_type == "flat":
            return 1
        elif bias_type == "front":
            return length - i - 1
        elif bias_type == "back":
            return i
        elif bias_type == "middle":
            if i <= length / 2:
                return i
            else:
                return length - i - 1
        else:
            raise ValueError(
                f"Invalid bias type: {bias_type}."
                "Should be from 'flat, 'front', 'middle', 'back'."
            )

    def gamma_select(self, cardinality, gamma: str, udf_gamma=None) -> float:
        """Select a gamma value based on the cardinality type."""
        if gamma == "one":
            return 1.0
        elif gamma == "reciprocal":
            if cardinality > 1:
                return 1 / cardinality
            else:
                return 1.0
        elif gamma == "udf_gamma":
            if udf_gamma is not None:
                return 1.0 / udf_gamma
            else:
                raise ValueError(
                    "udf_gamma must be provided for 'udf_gamma' gamma type."
                )
        else:
            raise ValueError("Invalid gamma type")

    def calculate_overlap_reward(self, y_pred, overlap_set, bias_type):
        """Overlap Reward for y_pred."""
        start, end = y_pred
        length = end - start + 1

        max_value = 0  # Total possible weighted value for all positions.
        my_value = 0  # Weighted value for overlapping positions only.

        for i in range(1, length + 1):
            global_position = start + i - 1
            bias_value = self.calculate_bias(i, length, bias_type)
            max_value += bias_value

            if global_position in overlap_set:
                my_value += bias_value

        return my_value / max_value if max_value > 0 else 0.0

    def ts_precision(
        self, y_pred, y_real, gamma="one", bias_type="flat", udf_gamma=None
    ):
        """Precision for either a single set or the entire time series."""
        #  Check if the input is a single set of predicted ranges or multiple sets
        is_single_set = isinstance(y_pred[0], tuple)
        """
        example:
        y_pred = [(1, 3), (5, 7)]
        y_real = [(2, 6), (8, 10)]
        """
        if is_single_set:
            # y_pred is a single set of predicted ranges
            total_overlap_reward = 0.0
            total_cardinality = 0

            for pred_range in y_pred:
                overlap_set = set()
                for real_start, real_end in y_real:
                    overlap_set.update(
                        range(
                            max(pred_range[0], real_start),
                            min(pred_range[1], real_end) + 1,
                        )
                    )

                overlap_reward = self.calculate_overlap_reward(
                    y_pred, overlap_set, bias_type
                )
                cardinality = len(overlap_set)
                gamma_value = self.gamma_select(cardinality, gamma, udf_gamma)

            total_overlap_reward += gamma_value * overlap_reward
            total_cardinality += 1  # Count each predicted range once

            return (
                total_overlap_reward / total_cardinality
                if total_cardinality > 0
                else 0.0
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
                precision = self.ts_precision(
                    pred_ranges, y_real, gamma, bias_type, udf_gamma
                )  # Recursive call for single sets
                total_precision += precision
                total_ranges += len(pred_ranges)

            return total_precision / total_ranges if total_ranges > 0 else 0.0
