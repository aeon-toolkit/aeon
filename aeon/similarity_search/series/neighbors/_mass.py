"""Implementation of NN with MASS."""

__maintainer__ = ["baraline"]
__all__ = ["MassSNN"]

import numpy as np
from numba import get_num_threads, njit, prange

from aeon.similarity_search.series._base import BaseSeriesSimilaritySearch
from aeon.similarity_search.series._commons import (
    _check_X_index,
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile,
    fft_sliding_dot_product,
)
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    sliding_mean_std_one_series,
)


class MassSNN(BaseSeriesSimilaritySearch):
    """
    Estimator to compute the subsequences nearest neighbors using MASS _[1].

    Parameters
    ----------
    length : int
        The length of the subsequences to use for the search.
    normalize : bool
        Whether the subsequences should be z-normalized.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `compute_distance_profile`.
        ``-1`` means using all processors.
        ``1`` means sequential computation (using FFT based implementation).
        Values > 1 uses a parallel implementation with sliding dot products.

    References
    ----------
    .. [1] Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy
    Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2015), The Fastest Similarity
    Search Algorithm for Time Series Subsequences under Euclidean Distance.
    """

    def __init__(
        self,
        length: int,
        normalize: bool | None = False,
        n_jobs: int = 1,
    ):
        self.normalize = normalize
        self.length = length
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        if self.normalize:
            self.X_means_, self.X_stds_ = sliding_mean_std_one_series(X, self.length, 1)
        return self

    def _predict(
        self,
        X: np.ndarray,
        k: int | None = 1,
        dist_threshold: float | None = np.inf,
        allow_trivial_matches: bool | None = False,
        exclusion_factor: float | None = 0.5,
        inverse_distance: bool | None = False,
        X_index: int | None = None,
    ):
        """
        Compute nearest neighbors to X in subsequences of X_.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, length)
            Subsequence we want to find neighbors for.
        k : int
            The number of neighbors to return.
        dist_threshold : float
            The maximum allowed distance of a candidate subsequence of X_ to X
            for the candidate to be considered as a neighbor.
        allow_trivial_matches: bool, optional
            Whether a neighbors of a match to a query can be also considered as matches
            (True), or if an exclusion zone is applied around each match to avoid
            trivial matches with their direct neighbors (False).
        inverse_distance : bool
            If True, the matching will be made on the inverse of the distance, and thus,
            the farther neighbors will be returned instead of the closest ones.
        exclusion_factor : float, default=1.
            A factor of the query length used to define the exclusion zone when
            ``allow_trivial_matches`` is set to False. For a given timestamp,
            the exclusion zone starts from
            :math:``id_timestamp - floor(length * exclusion_factor)`` and end at
            :math:``id_timestamp + floor(length * exclusion_factor)``.
        X_index : int, optional
            If ``X`` is a subsequence of X_, specify its starting timestamp in ``X_``.
            If specified, neighboring subsequences of X won't be able to match as
            neighbors.

        Returns
        -------
        np.ndarray, shape = (k)
            The indexes of the best matches in ``distance_profile``.
        np.ndarray, shape = (k)
            The distances of the best matches.

        """
        if X.shape[1] != self.length:
            raise ValueError(
                f"Expected X to have {self.length} timepoints but"
                f" got {X.shape[1]} timepoints."
            )
        X_index = _check_X_index(X_index, self.n_timepoints_, self.length)
        dist_profile = self.compute_distance_profile(X)
        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)

        exclusion_size = int(self.length * exclusion_factor)
        if X_index is not None:
            _max_timestamp = self.n_timepoints_ - self.length
            ub = min(X_index + exclusion_size, _max_timestamp)
            lb = max(0, X_index - exclusion_size)
            dist_profile[lb:ub] = np.inf

        if k == np.inf:
            k = len(dist_profile)

        return _extract_top_k_from_dist_profile(
            dist_profile,
            k,
            dist_threshold,
            allow_trivial_matches,
            exclusion_size,
        )

    def compute_distance_profile(self, X: np.ndarray):
        """
        Compute the distance profile of X to all samples in X_.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, length)
            The query to use to compute the distance profiles.

        Returns
        -------
        distance_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
            The distance profile of X to all samples in X_. The ``n_candidates`` value
            is equal to ``n_timepoins - length + 1``. If X_ is an unequal length
            collection, returns a numba typed list instead of an ndarray.

        """
        if self.n_jobs != 1:
            if self.normalize:
                _X_means = self.X_means_
                _X_stds = self.X_stds_
            else:
                _X_means = np.zeros((1, 1))
                _X_stds = np.zeros((1, 1))

            return _compute_distance_profile_parallel(
                self.X_,
                X,
                self.length,
                _X_means,
                _X_stds,
                self.normalize,
                self.n_jobs,
            )

        QT = fft_sliding_dot_product(self.X_, X)

        if self.normalize:
            distance_profile = _normalized_squared_distance_profile(
                QT,
                self.X_means_,
                self.X_stds_,
                X.mean(axis=1),
                X.std(axis=1),
                self.length,
            )
        else:
            distance_profile = _squared_distance_profile(
                QT,
                self.X_,  # T
                X,  # Q
            )

        return distance_profile

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
        """
        if parameter_set == "default":
            params = {"length": 20}
        else:
            raise NotImplementedError(
                f"The parameter set {parameter_set} is not yet implemented"
            )
        return params


@njit(cache=True, fastmath=True)
def _squared_distance_profile(QT, T, Q):
    """
    Compute squared Euclidean distance profile between query and a time series.

    This function calculates the squared distance profile for a single time series by
    leveraging the dot product of the query and time series as well as precomputed sums
    of squares to efficiently compute the squared distances.

    Parameters
    ----------
    QT : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The dot product between the query and the time series.
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    Q : np.ndarray
        2D array of shape (n_channels, query_length) representing query subsequence.

    Returns
    -------
    distance_profile : np.ndarray
        2D array of shape (n_channels, n_timepoints - query_length + 1)
        The squared distance profile between the query and the input time series.
    """
    n_channels, profile_length = QT.shape
    query_length = Q.shape[1]
    _QT = -2 * QT
    distance_profile = np.zeros(profile_length)
    for k in range(n_channels):
        _sum = 0
        _qsum = 0
        for j in range(query_length):
            _sum += T[k, j] ** 2
            _qsum += Q[k, j] ** 2

        distance_profile += _qsum + _QT[k]
        distance_profile[0] += _sum
        for i in range(1, profile_length):
            _sum += T[k, i + (query_length - 1)] ** 2 - T[k, i - 1] ** 2
            distance_profile[i] += _sum
    return distance_profile


@njit(cache=True, fastmath=True)
def _normalized_squared_distance_profile(
    QT, T_means, T_stds, Q_means, Q_stds, query_length
):
    """
    Compute the z-normalized squared Euclidean distance profile for one time series.

    Parameters
    ----------
    QT : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The dot product between the query and the time series.
    T_means : np.ndarray, 1D array of length n_channels
        The mean values of the time series for each channel.
    T_stds : np.ndarray, 2D array of shape (n_channels, profile_length)
        The standard deviations of the time series for each channel and position.
    Q_means : np.ndarray, 1D array of shape (n_channels)
        Means of the query q
    Q_stds : np.ndarray, 1D array of shape (n_channels)
        Stds of the query q
    query_length : int
        The length of the query subsequence used for the distance profile computation.


    Returns
    -------
    np.ndarray
        2D array of shape (n_channels, n_timepoints - query_length + 1) containing the
        z-normalized squared distance profile between the query subsequence and the time
        series. Entries are computed based on the z-normalized values, with special
        handling for constant values.
    """
    n_channels, profile_length = QT.shape
    distance_profile = np.zeros(profile_length)
    Q_is_constant = Q_stds <= AEON_NUMBA_STD_THRESHOLD
    for i in range(profile_length):
        Sub_is_constant = T_stds[:, i] <= AEON_NUMBA_STD_THRESHOLD
        for k in range(n_channels):
            # Two Constant case
            if Q_is_constant[k] and Sub_is_constant[k]:
                _val = 0
            # One Constant case
            elif Q_is_constant[k] or Sub_is_constant[k]:
                _val = query_length
            else:
                denom = query_length * Q_stds[k] * T_stds[k, i]

                p = (QT[k, i] - query_length * (Q_means[k] * T_means[k, i])) / denom
                p = min(p, 1.0)

                _val = abs(2 * query_length * (1.0 - p))
            distance_profile[i] += _val

    return distance_profile


@njit(cache=True, fastmath=True)
def _sliding_dot_product(T, Q, QT_buffer):
    n_channels, series_len = T.shape
    n_channels, query_len = Q.shape
    out_len = series_len - query_len + 1

    for k in range(n_channels):
        for i in range(out_len):
            dot = 0.0
            for j in range(query_len):
                dot += T[k, i + j] * Q[k, j]
            QT_buffer[k, i] = dot


@njit(cache=True, parallel=True)
def _compute_distance_profile_parallel(
    T, Q, query_length, T_means, T_stds, normalize, n_jobs
):
    n_channels = T.shape[0]
    n_timepoints = T.shape[1]
    if n_jobs < 1:
        n_jobs = get_num_threads()
    out_len = n_timepoints - query_length + 1
    chunk_size = int(np.ceil(out_len / n_jobs))
    final_profile = np.empty(out_len)
    if normalize:
        Q_means = np.empty(n_channels)
        Q_stds = np.empty(n_channels)
        for k in range(n_channels):
            q_sum = 0.0
            q_sq_sum = 0.0
            for j in range(query_length):
                val = Q[k, j]
                q_sum += val
                q_sq_sum += val * val
            Q_means[k] = q_sum / query_length
            Q_stds[k] = np.sqrt((q_sq_sum / query_length) - (Q_means[k] ** 2))
    else:
        Q_means = np.zeros(1)
        Q_stds = np.zeros(1)

    for i in prange(n_jobs):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, out_len)
        if start >= end:
            continue
        t_start = start
        t_end = end + query_length - 1
        T_chunk = T[:, t_start:t_end]
        chunk_out_len = end - start
        QT_chunk = np.empty((n_channels, chunk_out_len))
        _sliding_dot_product(T_chunk, Q, QT_chunk)

        if normalize:
            T_means_chunk = T_means[:, start:end]
            T_stds_chunk = T_stds[:, start:end]
            chunk_dist = _normalized_squared_distance_profile(
                QT_chunk,
                T_means_chunk,
                T_stds_chunk,
                Q_means,
                Q_stds,
                query_length,
            )
        else:
            chunk_dist = _squared_distance_profile(QT_chunk, T_chunk, Q)
        final_profile[start:end] = chunk_dist

    return final_profile
