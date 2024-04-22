"""MERLIN anomaly detector."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["MERLIN"]

import warnings

import numpy as np
from numba import njit

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.distances import squared_distance
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD
from aeon.utils.numba.stats import mean, std


class MERLIN(BaseAnomalyDetector):
    """MERLIN anomaly detector.

    MERLIN is a discord discovery algorithm that uses a sliding window to find the
    most anomalous subsequence in a time series. The algorithm is based on the
    Euclidean distance between subsequences of the time series.

    Parameters
    ----------
    min_length : int, default=5
        Minimum length of the subsequence to search for. Must be at least 4.
    max_length : int, default=50
        Maximum length of the subsequence to search for. Must be at half the length
        of the time series or less.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection import MERLIN
    >>> X = np.array([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4])
    >>> detector = MERLIN(min_length=4, max_length=5)
    >>> detector.fit_predict(X)
    array([False, False, False, False,  True,  True, False, False, False,
           False, False, False, False, False, False, False])
    """

    def __init__(self, min_length=5, max_length=50):
        self.min_length = min_length
        self.max_length = max_length

        super().__init__(axis=1)

    def _predict(self, X):
        X = X.squeeze()

        if X.shape[0] < self.min_length:
            raise ValueError(
                f"Series length of X {X.shape[0]} is less than min_length "
                f"{self.min_length}"
            )
        elif self.min_length > self.max_length:
            raise ValueError(
                f"min_length {self.min_length} must be less than max_length "
                f"{self.max_length}"
            )
        elif self.min_length < 4:
            raise ValueError("min_length must be at least 4")
        elif int(len(X) / 2) < self.max_length:
            raise ValueError(
                f"Series length of X {X.shape[0]} must be at least double max_length "
                f"{self.max_length}"
            )

        for i in range(X.shape[0] - self.min_length + 1):
            if std(X[i : i + self.min_length]) > AEON_NUMBA_STD_THRESHOLD:
                warnings.warn(
                    "There is region close to constant that will cause the results "
                    "to be unstable. It is suggested to delete the constant region "
                    "or try again with a longer min_length.",
                    stacklevel=2,
                )

        lengths = np.linspace(
            self.min_length,
            self.max_length,
            num=self.max_length - self.min_length + 1,
            dtype=np.int32,
        )

        r = 2 * np.sqrt(self.min_length)
        distances = np.full(len(lengths), -1.0)
        indicies = np.full(len(lengths), -1)
        while distances[0] < 0:
            indicies[0], distances[0] = self._drag(X, lengths[0], r)
            r = r * 0.5

        for i in range(1, min(5, len(lengths))):
            r = distances[i - 1] * 0.99
            while distances[i] < 0:
                indicies[i], distances[i] = self._drag(X, lengths[i], r)
                r = r * 0.99

        for i in range(min(5, len(lengths)), len(lengths)):
            m = mean(distances[i - 5 : i])
            s = std(distances[i - 5 : i])
            r = m - 2 * s
            indicies[i], distances[i] = self._drag(X, lengths[i], r)
            while distances[i] < 0:
                indicies[i], distances[i] = self._drag(X, lengths[i], r)
                r = r - s

        if np.all(distances == -1):
            raise ValueError("No discord found in the series.")

        anomalies = np.zeros(X.shape[0], dtype=bool)
        for i in indicies:
            if i != np.nan:
                anomalies[i] = True

        return anomalies

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _drag(X, length, discord_range):
        C = []
        data = np.zeros((X.shape[0] - length + 1, length))
        for i in range(X.shape[0] - length + 1):
            is_candidate = True
            data[i] = X[i : i + length]
            sstd = std(data[i]) + AEON_NUMBA_STD_THRESHOLD
            data[i] = (data[i] - mean(data[i])) / sstd

            for n, j in enumerate(C):
                if (
                    np.abs(i - j) >= length
                    and squared_distance(data[i], data[j]) < discord_range
                ):
                    del C[n]
                    is_candidate = False
                    break

            if is_candidate:
                C.append(i)

        if len(C) == 0:
            return -1, -1

        D = [np.inf] * len(C)
        del_list = [False] * len(C)
        for i in range(X.shape[0] - length + 1):
            for n, j in enumerate(C):
                if np.abs(i - j) >= length:
                    d = squared_distance(data[i], data[j])
                    if d < discord_range:
                        del_list[n] = True
                    else:
                        D[n] = np.minimum(D[n], d)

            for n, j in enumerate(del_list):
                if j:
                    del C[n]
                    del D[n]
                    del del_list[n]

        if len(C) == 0:
            return -1, -1

        all_inf = True
        for n in range(len(C)):
            if D[n] == np.inf:
                D[n] = -1
            else:
                all_inf = False

        if all_inf:
            return np.nan, np.nan

        d_max = int(np.argmax(np.array(D)))
        return C[d_max], np.sqrt(D[d_max])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {"min_length": 4, "max_length": 7}
