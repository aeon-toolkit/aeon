"""MADRID anomaly detector. based on (https://sites.google.com/view/madrid-icdm-23/home)"""

__mentainer__ = ["acquayefrank"]
__all__ = ["MADRID"]

import numpy as np
import math

from aeon.anomaly_detection.base import BaseAnomalyDetector


class MADRID(BaseAnomalyDetector):

    def __init__(
        self,
        min_length,
        max_length,
        step_size=1,
        split_psn=None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0")
        if split_psn:
            self.split_psn = split_psn
        else:
            self.split_psn = (
                self.max_length + 1
            )  # should be greater than self.max_length
        self.step_size = step_size
        super().__init__(axis=1)

    def _predict(self, X, factor=1) -> np.ndarray:
        if X.shape[0] < self.min_length:
            raise ValueError(
                f"Series length of X {X.shape[0]} is less than min_length "
                f"{self.min_length}"
            )

        if self._contains_constant_regions(X, self.min_length):
            error_message = (
                "BREAK: There is at least one region of length min_length that is constant, or near constant.\n\n"
                "To fix this issue:\n"
                "1) Choose a longer length for min_length.\n"
                "2) Add a small amount of noise to the entire time series (this will probably result in the current constant sections becoming top discords).\n"
                "3) Add a small linear trend to the entire time series (this will probably result in the current constant sections becoming motifs, and not discords).\n"
                "4) Carefully edit the data to remove the constant sections."
            )
            raise ValueError(error_message)

        X = X.squeeze()

        if factor in [1, 2, 4, 8, 16]:
            if (self.max_length + factor - 1) // factor < 2:
                raise ValueError(
                    f"MADRID cannot be executed properly because {self.min_length}/{factor} < 2, please select an option again"
                )
            if (self.max_length + factor - 1) // factor < 2:
                raise ValueError(
                    f"MADRID cannot be executed properly because {self.max_length}/{factor} < 2, please select an option again"
                )
        else:
            raise ValueError(
                f"factor must be one of these numbers [1, 2, 4, 8, 16] in order to compute MADRID."
            )
        self._compute_madrid(X)

    def _contains_constant_regions(self, X, sub_sequence_length):
        bool_vec = False  # in the origianl matlab code they use 0,1 but trus, false is a better representation
        X = np.asarray(X)

        constant_indices = np.where(np.diff(X) != 0)[0] + 1
        constant_indices = np.concatenate(([0], constant_indices, [len(X)]))
        constant_lengths = np.diff(constant_indices)

        constant_length = max(constant_lengths)

        if constant_length >= sub_sequence_length or np.var(X) < 0.2:
            bool_vec = True

        return bool_vec

    def _dump_2_0(
        self,
        X,
    ):
        pass

    def estimate_factor(self, X):
        len_x = len(X)
        factor = 1

        if (
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            )
            < 5000000
        ):
            # polynomial model (of order 6)
            p_1 = [
                -4.66922312132205e-45,
                1.54665628995475e-35,
                -1.29314859463985e-26,
                2.01592418847342e-18,
                -2.54253065977245e-11,
                9.77027495487874e-05,
                -1.27055582771851e-05,
            ]
            p_2 = [
                -3.79100071825804e-42,
                3.15547030055575e-33,
                -6.62877819318290e-25,
                2.59437174380763e-17,
                -8.10970871564789e-11,
                7.25344313152170e-05,
                4.68415490390476e-07,
            ]
        else:
            # linear model
            p_1 = [3.90752957831437e-05, 0]
            p_2 = [1.94005690535588e-05, 0]

        p_4 = [1.26834880558841e-05, 0]
        p_8 = [1.42210521045333e-05, 0]
        p_16 = [1.82290885539705e-05, 0]

        # Prediction
        factor = 16
        predicted_execution_time_16 = np.polyval(
            p_16,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 8
        predicted_execution_time_8 = np.polyval(
            p_8,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 4
        predicted_execution_time_4 = np.polyval(
            p_4,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 2
        predicted_execution_time_2 = np.polyval(
            p_2,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        factor = 1
        predicted_execution_time_1 = np.polyval(
            p_1,
            len(range(1, len_x - self.split_psn, factor))
            * len(
                range(
                    math.ceil(self.min_length / factor),
                    math.ceil(self.max_length / factor),
                    self.step_size,
                )
            ),
        )

        if predicted_execution_time_1 < 10:
            factor = 1
            print(
                f"Predicted execution time for MADRID is small, hence potential factor choise is 1"
            )
        else:
            test_data = self._test_data_madrid()
            factor = 16
            actual_measurement_16 = self._compute_madrid(test_data)
            scaling_factor = actual_measurement_16 / 0.65461
            predicted_execution_time_16_scaled = (
                predicted_execution_time_16 * scaling_factor
            )
            predicted_execution_time_8_scaled = (
                predicted_execution_time_8 * scaling_factor
            )
            predicted_execution_time_4_scaled = (
                predicted_execution_time_4 * scaling_factor
            )
            predicted_execution_time_2_scaled = (
                predicted_execution_time_2 * scaling_factor
            )
            predicted_execution_time_1_scaled = (
                predicted_execution_time_1 * scaling_factor
            )

            print(
                f"Predicted execution time for MADRID 1 to 16: {predicted_execution_time_16_scaled:.1f} seconds, hence potential factor choise is 16"
            )
            print(
                f"Predicted execution time for MADRID 1 to 8: {predicted_execution_time_8_scaled:.1f} seconds, hence potential factor choise is 8"
            )
            print(
                f"Predicted execution time for MADRID 1 to 4: {predicted_execution_time_4_scaled:.1f} seconds, hence potential factor choise is 4"
            )
            print(
                f"Predicted execution time for MADRID 1 to 2: {predicted_execution_time_2_scaled:.1f} seconds, hence potential factor choise is 2"
            )
            print(
                f"Predicted execution time for MADRID 1 to 1: {predicted_execution_time_1_scaled:.1f} seconds, hence potential factor choise is 1"
            )

    def _compute_madrid(self, X):
        bfs_seed = float("-inf")  # used for first time run of dump_topk
        k = 1
        time_bf = 0
        num_rows = int(
            np.ceil((self.max_length + 1 - self.min_length) / self.step_size)
        )
        num_cols = len(X)
        multilength_discord_table = np.full((num_rows, num_cols), -np.inf)
        bsf = np.zeros((num_rows, 1))
        bsf_loc = np.full((num_rows, 1), np.nan)

        m_set = np.arange(self.min_length, self.max_length + 1, self.step_size)
        m_pointer = int(np.ceil(len(m_set) / 2))
        m = m_set[m_pointer]

        # anomalies = np.zeros(X.shape[0], dtype=bool)
        # return anomalies

    def _test_data_madrid(self):
        np.random.seed(123456789)

        Fs = 10000
        t = np.arange(0, 10 + 1 / Fs, 1 / Fs)  # Time vector
        f_in_start = 50
        f_in_end = 60
        f_in = np.linspace(f_in_start, f_in_end, len(t))  # Linearly spaced frequency
        phase_in = np.cumsum(f_in / Fs)  # Cumulative phase

        y = np.sin(2 * np.pi * phase_in)  # Sinusoidal signal
        y += np.random.randn(len(y)) / 12  # Add noise

        EndOfTrain = len(y) // 2  # End of training segment

        # Add medium anomaly
        anomaly_start_1 = EndOfTrain + 1200
        y[anomaly_start_1 : anomaly_start_1 + 64] += np.random.randn(64) / 3

        # Add another anomaly
        anomaly_start_2 = EndOfTrain + 4180
        y[anomaly_start_2 : anomaly_start_2 + 160] += np.random.randn(160) / 4

        # Add long duration anomaly
        anomaly_start_3 = EndOfTrain + 8200
        anomaly_end_3 = EndOfTrain + 8390
        y[anomaly_start_3:anomaly_end_3] *= 0.5

        return y
