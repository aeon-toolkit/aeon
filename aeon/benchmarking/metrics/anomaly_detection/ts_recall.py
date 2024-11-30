"""Calculate the Recall metric of a time series anomaly detection model."""


class RangeRecall:
    """Calculates Recall for time series.

    Parameters
    ----------
    y_real : np.ndarray
        Set of ground truth anomaly ranges (actual anomalies).

    y_pred : np.ndarray
        Set of predicted anomaly ranges.

    cardinality : str, default="one"
        Number of overlaps between y_pred and y_real.

    gamma : float, default=1.0
        Overlap Cardinality Factor. Penalizes or adjusts the metric based on
        the cardinality.
        Should be one of {'reciprocal', 'one', 'udf_gamma'}.

    alpha : float
        Weight of the existence reward. Since Recall emphasizes coverage,
        you might adjust this value if needed.

    bias : str, default="flat"
        Captures the importance of positional factors within anomaly ranges.
        Determines the weight given to specific portions of anomaly range
        when calculating overlap rewards.
        Should be one of {'flat', 'front', 'middle', 'back'}.

    omega : float
        Measure the extent and overlap between y_pred and y_real.
        Considers the size and position of overlap and rewards. Should
        be a float value between 0 and 1.
    """

    def _init_(self, bias="flat", alpha=0.0, gamma="one"):
        assert gamma in ["reciprocal", "one", "udf_gamma"], "Invalid gamma type"
        assert bias in ["flat", "front", "middle", "back"], "Invalid bias type"

        self.bias = bias
        self.alpha = alpha
        self.gamma = gamma

    def calculate_bias(self, position, length, bias_type="flat"):
        """Calculate bias value based on position and length.

        Args:
            position: Current position in the range
            length: Total length of the range
            bias_type: Type of bias to apply (default: "flat")
        """
        if bias_type == "flat":
            return 1.0
        elif bias_type == "front":
            return 1.0 - (position - 1) / length
        elif bias_type == "middle":
            return (
                1.0 - abs(2 * (position - 1) / (length - 1) - 1) if length > 1 else 1.0
            )
        elif bias_type == "back":
            return position / length
        else:
            return 1.0

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

    def calculate_overlap_reward_recall(self, real_range, overlap_set, bias_type):
        """Overlap Reward for y_real."""
        start, end = real_range
        length = end - start + 1

        max_value = 0.0  # Total possible weighted value for all positions.
        my_value = 0.0  # Weighted value for overlapping positions only.

        for i in range(1, length + 1):
            global_position = start + i - 1
            bias_value = self.calculate_bias(i, length, bias_type)
            max_value += bias_value

            if global_position in overlap_set:
                my_value += bias_value

        return my_value / max_value if max_value > 0 else 0.0

    def ts_recall(self, y_pred, y_real, gamma="one", bias_type="flat", udf_gamma=None):
        """Calculate Recall for time series anomaly detection."""
        if isinstance(y_real[0], tuple):
            total_overlap_reward = 0.0
            total_cardinality = 0

            for real_range in y_real:
                overlap_set = set()
                cardinality = 0

                for pred_start, pred_end in y_pred:
                    overlap_start = max(real_range[0], pred_start)
                    overlap_end = min(real_range[1], pred_end)

                    if overlap_start <= overlap_end:
                        overlap_set.update(range(overlap_start, overlap_end + 1))
                        cardinality += 1

                if overlap_set:
                    overlap_reward = self.calculate_overlap_reward_recall(
                        real_range, overlap_set, bias_type
                    )
                    gamma_value = self.gamma_select(cardinality, gamma, udf_gamma)

                    total_overlap_reward += gamma_value * overlap_reward
                    total_cardinality += 1

            return total_overlap_reward / len(y_real) if y_real else 0.0

        # Handle multiple sets of y_real
        elif (
            isinstance(y_real, list) and len(y_real) > 0 and isinstance(y_real[0], list)
        ):
            total_recall = 0.0
            total_real = 0

            for real_ranges in y_pred:  # Iterate over all sets of real ranges
                precision = self.ts_recall(
                    real_ranges, y_real, gamma, bias_type, udf_gamma
                )
                total_recall += precision * len(real_ranges)
                total_real += len(real_ranges)

            return total_recall / total_real if total_real > 0 else 0.0
