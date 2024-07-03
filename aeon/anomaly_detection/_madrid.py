"""MADRID anomaly detector. based on (https://sites.google.com/view/madrid-icdm-23/home)"""

__mentainer__ = ["acquayefrank"]
__all__ = ["MADRID"]

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector


class MADRID(BaseAnomalyDetector):

    def __init__(self, min_length):
        self.min_length = min_length

        super().__init__(axis=1)

    def _predict(self, X) -> np.ndarray:
        X = X.squeeze()
        if X.shape[0] < self.min_length:
            raise ValueError(
                f"Series length of X {X.shape[0]} is less than min_length "
                f"{self.min_length}"
            )

        # TODO continue from here.
        print(self._contains_constant_regions(X, self.min_length))

        anomalies = np.zeros(X.shape[0], dtype=bool)
        return anomalies

    def _contains_constant_regions(self, T, sub_sequence_length):
        bool_vec = False  # in the origianl matlab code they use 0,1 but trus, false is a better representation
        T = np.asarray(T)

        constant_indices = np.where(np.diff(T) != 0)[0] + 1
        constant_indices = np.concatenate(([0], constant_indices, [len(T)]))
        constant_lengths = np.diff(constant_indices)

        constant_length = max(constant_lengths)

        if constant_length >= sub_sequence_length or np.var(T) < 0.2:
            bool_vec = True

        return bool_vec
