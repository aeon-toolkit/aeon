"""Implementation of NN with MASS."""

__maintainer__ = ["baraline"]
__all__ = ["MASS"]

import numpy as np
from numba import njit

from aeon.similarity_search.subsequence._base import BaseSubsequenceSearch
from aeon.similarity_search.subsequence._commons import (
    _inverse_distance_profile,
    extract_top_k_from_dist_profiles_2d,
    fft_sliding_dot_product,
)
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    sliding_mean_std_one_series,
)


class MASS(BaseSubsequenceSearch):
    """
    Subsequence nearest neighbor search using MASS algorithm.

    MASS (Mueen's Algorithm for Similarity Search) originally computes the distance
    profile between a query subsequence and all subsequences in a time series using
    FFT-based convolution. This estimator adapts it to search for the k nearest
    neighbor subsequences across a collection and returns the best matches with their
    ``(case_index, timestamp)`` locations.

    Parameters
    ----------
    length : int
        The length of the subsequences to use for the search. The query provided
        to ``predict`` must have exactly this many timepoints.
    normalize : bool, default=False
        Whether the subsequences should be z-normalized before distance computation.
        This results in scale-independent matching, useful when you want to find
        patterns regardless of their amplitude.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The fitted collection of time series.
    n_cases_ : int
        Number of time series in the fitted collection.
    n_channels_ : int
        Number of channels in the fitted time series.
    n_timepoints_ : int
        Number of timepoints in each fitted time series.

    See Also
    --------
    BruteForce : Brute force subsequence search (slower but simpler).

    References
    ----------
    .. [1] Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy
       Viswanathan, Chetan Kumar Gupta and Eamonn Keogh (2015), The Fastest Similarity
       Search Algorithm for Time Series Subsequences under Euclidean Distance.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.subsequence import MASS
    >>> X_fit = np.random.rand(5, 1, 100)
    >>> query = np.random.rand(1, 20)
    >>> searcher = MASS(length=20, normalize=False)
    >>> searcher.fit(X_fit)
    >>> indexes, distances = searcher.predict(query, k=3)
    """

    _tags = {
        "capability:unequal_length": False,
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        length: int,
        normalize: bool | None = False,
    ):
        self.normalize = normalize
        super().__init__(length)

    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        """
        Fit the MASS estimator on a collection of time series.

        Parameters
        ----------
        X : np.ndarray, shape=(n_cases, n_channels, n_timepoints)
            Collection of time series to search within.
        y : ignored

        Returns
        -------
        self
        """
        # Precompute means and stds for each series in the collection
        if self.normalize:
            n_cases = X.shape[0]
            # Store means and stds for each series
            self.X_means_ = []
            self.X_stds_ = []
            for i in range(n_cases):
                means, stds = sliding_mean_std_one_series(X[i], self.length, 1)
                self.X_means_.append(means)
                self.X_stds_.append(stds)
        return self

    def _predict(
        self,
        X: np.ndarray,
        k: int = 1,
        dist_threshold: float = np.inf,
        allow_trivial_matches: bool = False,
        exclusion_factor: float = 0.5,
        inverse_distance: bool = False,
        X_index: tuple = None,
    ):
        """
        Find nearest neighbor subsequences to X in the fitted collection.

        Parameters
        ----------
        X : np.ndarray, shape=(n_channels, length)
            Query subsequence.
        k : int, default=1
            Number of neighbors to return.
        dist_threshold : float, default=np.inf
            Maximum distance threshold for matches.
        allow_trivial_matches : bool, default=False
            Whether to allow neighboring matches within the same series.
        exclusion_factor : float, default=0.5
            Factor of query length for exclusion zone size.
        inverse_distance : bool, default=False
            If True, return farthest neighbors instead.
        X_index : tuple (i_case, i_timepoint), optional
            If X is from the fitted collection, specify its location.

        Returns
        -------
        indexes : np.ndarray, shape=(n_matches, 2)
            The (i_case, i_timepoint) indexes of the best matches.
        distances : np.ndarray, shape=(n_matches,)
            The distances of the best matches.
        """
        self._check_query_length(X)

        dist_profiles = self.compute_distance_profile(X)

        if inverse_distance:
            for i in range(len(dist_profiles)):
                dist_profiles[i] = _inverse_distance_profile(dist_profiles[i])

        exclusion_size = int(self.length * exclusion_factor)

        if X_index is not None:
            i_case, i_timepoint = X_index
            if i_case < 0 or i_case >= self.n_cases_:
                raise ValueError(
                    f"X_index case {i_case} is out of bounds for collection "
                    f"with {self.n_cases_} cases."
                )
            _max_timestamp = self.n_timepoints_ - self.length
            ub = min(i_timepoint + exclusion_size, _max_timestamp)
            lb = max(0, i_timepoint - exclusion_size)
            dist_profiles[i_case, lb:ub] = np.inf

        return extract_top_k_from_dist_profiles_2d(
            dist_profiles,
            k,
            dist_threshold,
            allow_trivial_matches,
            exclusion_size,
        )

    def compute_distance_profile(self, X: np.ndarray):
        """
        Compute the distance profile of X to all subsequences in X_.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, length)
            The query to use to compute the distance profiles.

        Returns
        -------
        distance_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
            The distance profile of X to all subsequences in all series of X_.
            The ``n_candidates`` value is equal to ``n_timepoints - length + 1``.
        """
        n_cases = self.X_.shape[0]
        n_candidates = self.n_timepoints_ - self.length + 1
        distance_profiles = np.zeros((n_cases, n_candidates))

        Q_means = X.mean(axis=1) if self.normalize else None
        Q_stds = X.std(axis=1) if self.normalize else None

        for i in range(n_cases):
            QT = fft_sliding_dot_product(self.X_[i], X)

            if self.normalize:
                distance_profiles[i] = _normalized_squared_distance_profile(
                    QT,
                    self.X_means_[i],
                    self.X_stds_[i],
                    Q_means,
                    Q_stds,
                    self.length,
                )
            else:
                distance_profiles[i] = _squared_distance_profile(
                    QT,
                    self.X_[i],
                    X,
                )

        return distance_profiles

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
