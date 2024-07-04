"""MADRID anomaly detector. based on (https://sites.google.com/view/madrid-icdm-23/home)"""

__mentainer__ = ["acquayefrank"]
__all__ = ["MADRID"]

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector


class MADRID(BaseAnomalyDetector):

    def __init__(self, min_length, max_length, step_size=1):
        self.min_length = min_length
        self.max_length = max_length
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0")
        self.step_size = step_size
        super().__init__(axis=1)

    def _predict(self, X) -> np.ndarray:
        X = X.squeeze()
        bfs_seed = float('-inf') # used for first time run of dump_topk
        k = 1
        time_bf = 0

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
        
        num_rows = int(np.ceil((self.max_length + 1 - self.min_length)/self.step_size))
        num_cols = len(X)
        multilength_discord_table = np.full((num_rows, num_cols), -np.inf)
        bsf = np.zeros((num_rows, 1))
        bsf_loc = np.full((num_rows, 1), np.nan)

        m_set = np.arange(self.min_length, self.max_length + 1, self.step_size)
        m_pointer = int(np.ceil(len(m_set) / 2))
        m = m_set[m_pointer]

        # anomalies = np.zeros(X.shape[0], dtype=bool)
        # return anomalies

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
    
    def _dump_2_0(self, X, ):
        pass
