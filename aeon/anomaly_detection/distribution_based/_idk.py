"""IDK2 anomaly detector."""

__maintainer__ = ["Ramana-Raja"]
__all__ = ["IDK2"]

from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.windowing import reverse_windowing


class IDK2(BaseAnomalyDetector):
    """IDK² and s-IDK² anomaly detector.

    The Isolation Distributional Kernel (IDK) is a data-dependent kernel for efficient
    anomaly detection, improving accuracy without explicit learning. Its extension,
    IDK², simplifies group anomaly detection, outperforming traditional methods in
    speed and effectiveness.This implementation is inspired by the Isolation
    Distributional Kernel (IDK) approach as detailed in Kai Ming Ting, Bi-Cun Xu,
    Takashi Washio, Zhi-Hua Zhou (2020) [1]_.

    This Anomaly Detector assumes the input time series is stationary,
    so trends should be removed prior to detection. IDK² is recommended
    for periodic time series,while s-IDK² is better suited for non-periodic
    cases.The attribute `original_output_` stores the raw anomaly scores before
    reverse-windowing is applied when width > 1 only. The use of s-IDK² or IDK²
    is determined by the parameter "sliding".

    Parameters
    ----------
    psi1 : int, default=8
         The number of samples randomly selected in each iteration to construct the
         feature map matrix during the first stage. This parameter determines the
         granularity of the first-stage feature representation. Higher values allow
         the model to capture more detailed data characteristics but increase
         computational complexity.
    psi2 : int, default=2
         The number of samples randomly selected in each iteration to construct
         the feature map matrix during the second stage. This parameter
         determines the granularity of the second-stage feature representation.
         Higher values allow the model to capture more detailed
         data characteristics but increase computational complexity.
    width : int, default=1
         The size of the sliding or fixed-width window used for anomaly detection.
         For fixed-width processing, this defines the length of each segment analyzed.
         In sliding window mode, it specifies the length of the window moving
         across the data. Width (referred to as `m` in the original paper)
         should be equal to or greater than the period of the time series.
    t : int, default=100
         The number of iterations (time steps) for random sampling to
         construct the feature maps. Each iteration generates a set of random samples,
         which contribute to the feature map matrix. Larger values improve the
         robustness of the feature maps but increase the runtime.
    sliding : bool, default=False
         Determines whether IDK² or s-IDK² is used for anomaly detection.
         If True, the model uses s-IDK²
         If False, the model uses IDK²
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        If `int`, random_state is the seed used by the random number generator;
        If `np.random.RandomState` instance,
        random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`

    Notes
    -----
    GitHub Repository:
         IsolationKernel/Codes: IDK Implementation for Time Series Data
         URL: https://github.com/IsolationKernel/Codes/tree/main/IDK/TS

    References
    ----------
    [1]_ Kai Ming Ting, Bi-Cun Xu, Takashi Washio, Zhi-Hua Zhou (2020)
         'Isolation Distributional Kernel: A New Tool for Kernel-Based
          Anomaly Detection',
          DOI: https://dl.acm.org/doi/10.1145/3394486.3403062

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection.distribution_based import IDK2
    >>> X = np.array([1, 2, 3, 4, 1, 2, 3, 3], dtype=np.float64)
    >>> ad_sliding = IDK2(psi1=8, psi2=4, width=3, sliding=True, random_state=1)
    >>> ad_sliding.fit_predict(X)
    array([0.11      , 0.1225    , 0.12666667, 0.135     , 0.12666667,
           0.11833333, 0.11      , 0.11      ])

    >>> import numpy as np
    >>> from aeon.anomaly_detection.distribution_based import IDK2
    >>> X = np.array([1, 2, 3, 4, 1, 2, 3, 3], dtype=np.float64)
    >>> ad_sliding = IDK2(psi1=4, psi2=2, width=3, random_state=1)
    >>> ad_sliding.fit_predict(X)
    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0. , 0. ])

    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:multithreading": False,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        psi1: int = 8,
        psi2: int = 2,
        width: int = 1,
        t: int = 100,
        sliding: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        self.psi1 = psi1
        self.psi2 = psi2
        self.width = width
        self.t = t
        self.sliding = sliding
        self.random_state = random_state
        super().__init__(axis=0)

    def _compute_point_to_sample(self, X, sample_indices):
        sample = X[sample_indices, :]
        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)

        sample2sample = point2sample[sample_indices, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = np.nan

        radius_list = np.nanmin(sample2sample, axis=1)
        min_dist_point2sample = np.argmin(point2sample, axis=1)

        return point2sample, radius_list, min_dist_point2sample

    def _generate_feature_map(self, X, psi, t, rng):
        feature_matrix = np.zeros((X.shape[0], t * psi), dtype=int)
        for time in range(t):
            sample_indices = rng.choice(len(X), size=psi, replace=False)
            point2sample, radius_list, min_dist_point2sample = (
                self._compute_point_to_sample(X, sample_indices)
            )

            min_point2sample_index = np.argmin(point2sample, axis=1)
            min_dist_point2sample = min_point2sample_index + time * psi
            point2sample_value = point2sample[
                range(len(feature_matrix)), min_point2sample_index
            ]
            ind = point2sample_value < radius_list[min_point2sample_index]
            feature_matrix[ind, min_dist_point2sample[ind]] = 1

        return feature_matrix

    def _compute_idk_score(self, X, psi, t, rng):
        point_fm_list = self._generate_feature_map(X=X, psi=psi, t=t, rng=rng)
        feature_mean_map = np.mean(point_fm_list, axis=0)
        return np.dot(point_fm_list, feature_mean_map) / t

    def _idk_fixed_window(self, X, rng):
        window_num = int(np.ceil(X.shape[0] / self.width))
        featuremap_count = np.zeros((window_num, self.t * self.psi1))
        onepoint_matrix = np.full((X.shape[0], self.t), -1)

        for time in range(self.t):
            sample_indices = rng.choice(X.shape[0], size=self.psi1, replace=False)
            point2sample, radius_list, min_dist_point2sample = (
                self._compute_point_to_sample(X, sample_indices)
            )

            for i in range(X.shape[0]):
                if (
                    point2sample[i][min_dist_point2sample[i]]
                    < radius_list[min_dist_point2sample[i]]
                ):
                    onepoint_matrix[i][time] = (
                        min_dist_point2sample[i] + time * self.psi1
                    )
                    featuremap_count[int(i / self.width)][onepoint_matrix[i][time]] += 1

        for i in range(window_num):
            featuremap_count[i] /= self.width
        isextra = X.shape[0] - (int)(X.shape[0] / self.width) * self.width
        if isextra > 0:
            featuremap_count[-1] /= isextra
            featuremap_count = np.delete(
                featuremap_count, [featuremap_count.shape[0] - 1], axis=0
            )

        return self._compute_idk_score(
            featuremap_count, psi=self.psi2, t=self.t, rng=rng
        )

    def _idk_square_sliding(self, X, rng):
        point_fm_list = self._generate_feature_map(
            X=X, psi=self.psi1, t=self.t, rng=rng
        )
        point_fm_list = np.insert(point_fm_list, 0, 0, axis=0)
        cumsum = np.cumsum(point_fm_list, axis=0)

        subsequence_fm_list = (cumsum[self.width :] - cumsum[: -self.width]) / float(
            self.width
        )

        return self._compute_idk_score(
            X=subsequence_fm_list, psi=self.psi2, t=self.t, rng=rng
        )

    def _predict(self, X: np.ndarray) -> np.ndarray:
        rng = check_random_state(self.random_state)
        if self.sliding:
            sliding_output = self._idk_square_sliding(X, rng)
            reversed_output = reverse_windowing(
                y=sliding_output,
                window_size=self.width,
                stride=1,
                reduction=np.nanmean,
            )
            return reversed_output
        elif self.width > 1:
            self.original_output_ = self._idk_fixed_window(X, rng)
            final_output = np.repeat(self.original_output_, self.width)
            final_output = np.pad(
                final_output, (0, len(X) % self.width), mode="constant"
            )
            return final_output
        else:
            return self._idk_fixed_window(X, rng)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "psi1": 8,
            "psi2": 2,
            "width": 1,
        }
