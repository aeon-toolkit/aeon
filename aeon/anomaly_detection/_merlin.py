__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["MERLIN"]

import warnings

import numpy as np
from numba import njit

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.distances import euclidean_distance
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD
from aeon.utils.numba.stats import mean, std


class MERLIN(BaseAnomalyDetector):
    def __init__(self, min_length=5, max_length=50):
        self.min_length = min_length
        self.max_length = max_length

        super().__init__()

    def _predict(self, X):
        if X.shape[1] < self.min_length:
            raise ValueError(
                f"Series length of X {X.shape[1]} is less than min_length "
                f"{self.min_length}"
            )

        for i in range(X.shape[1] - self.min_length + 1):
            if std(X[:, i : i + self.min_length]) > AEON_NUMBA_STD_THRESHOLD:
                warnings.warn(
                    "There is region close to constant that will cause the results "
                    "will be unstable. It is suggested to delete the constant region "
                    "or try again with a longer min_length."
                )

        lengths = np.linspace(self.min_length, self.max_length, dtype=np.int32)

        r = 2 * np.sqrt(self.min_length)
        distances = np.full(len(lengths), -1)
        indicies = np.full(len(lengths), -1)
        while distances[0] < 0:
            indicies[0], distances[0] = self._drag(X, lengths[0], r)
            r = r * 0.5

        for i in range(1, 5):
            r = distances[i - 1] * 0.99
            while distances[i] < 0:
                indicies[i], distances[i] = self._drag(X, lengths[i], r)
                r = r * 0.99

        for i in range(5, len(lengths)):
            m = mean(distances[i - 5 : i])
            s = std(distances[i - 5 : i])
            r = m - 2 * s
            indicies[i], distances[i] = self._drag(X, lengths[i], r)
            while distances[i] < 0:
                indicies[i], distances[i] = self._drag(X, lengths[i], r)
                r = r - s

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _drag(X, length, discord_range):
        C = []
        data = np.zeros((X.shape[1] - length + 1, length))
        for i in range(X.shape[1] - length + 1):
            is_candidate = True
            data[i] = X[i : i + length]
            data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])

            for n, j in reversed(list(enumerate(C))):
                if (
                    np.abs(i - j) >= length
                    and euclidean_distance(data[i], data[j]) < discord_range
                ):
                    del C[n]
                    is_candidate = False

            if is_candidate:
                C.append(i)

        if len(C) == 0:
            return -1, -1

        D = [np.inf] * len(C)
        for i in range(X.shape[1] - length + 1):
            for n, j in reversed(list(enumerate(C))):
                if np.abs(i - j) >= length:
                    d = euclidean_distance(data[i], data[j])
                    if d < discord_range:
                        del C[n]
                        del D[n]
                    else:
                        D[n] = np.minimum(D[n], d)

        if len(C) == 0:
            return -1, -1

        max = int(np.argmax(D))
        return C[max] + int(length / 2), np.sqrt(D[max])
