"""Implementation of NN with MASS."""

__maintainer__ = ["baraline"]
__all__ = ["MASS"]

import numpy as np
import scipy.fft
from numba import njit

from aeon.similarity_search.subsequence._base import BaseDistanceProfileSearch
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    sliding_mean_std_one_series,
)


class MASS(BaseDistanceProfileSearch):
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
    NaiveSubsequenceSearch : Naive subsequence search (slower but simpler).

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
    MASS(length=20)
    >>> indexes, distances = searcher.predict(query, k=3)
    """

    _tags = {
        "capability:unequal_length": False,
        "capability:multivariate": True,
        "capability:multithreading": False,
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
        n_cases, n_channels, n_timepoints = X.shape

        # Precompute the FFT spectra of every fitted series once. These depend
        # only on the fitted data, so caching them here avoids recomputing a
        # forward FFT of each (long) series on every predict call. ``fft_len``
        # is sized for the full linear convolution of a length-``length`` query
        # with each series (``next_fast_len`` for FFT efficiency).
        self._fft_len_ = scipy.fft.next_fast_len(n_timepoints + self.length - 1)
        self.X_spectra_ = scipy.fft.rfft(X, n=self._fft_len_, axis=2)

        # Precompute means and stds for each series in the collection
        if self.normalize:
            # Store means and stds for each series
            self.X_means_ = []
            self.X_stds_ = []
            for i in range(n_cases):
                means, stds = sliding_mean_std_one_series(X[i], self.length, 1)
                self.X_means_.append(means)
                self.X_stds_.append(stds)
        else:
            # Precompute the sliding sum-of-squares of every fitted series. In
            # the non-normalized distance profile this term depends only on the
            # fitted data, so caching it avoids recomputing it on every predict.
            self.X_ssqs_ = _sliding_sum_of_squares(X, self.length)
        return self

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

        # Sliding dot products between the query and every fitted series, in a
        # single batched FFT. The reversed query is transformed once per
        # channel, broadcast-multiplied against the cached series spectra, and
        # inverse-transformed in one batched call, reusing the FFT work cached
        # at fit time instead of recomputing a per-series FFT on every call.
        QT_all = _batched_sliding_dot_product(
            X, self.X_spectra_, self._fft_len_, self.n_timepoints_
        )

        Q_means = X.mean(axis=1) if self.normalize else None
        Q_stds = X.std(axis=1) if self.normalize else None

        for i in range(n_cases):
            if self.normalize:
                distance_profiles[i] = _normalized_squared_distance_profile(
                    QT_all[i],
                    self.X_means_[i],
                    self.X_stds_[i],
                    Q_means,
                    Q_stds,
                    self.length,
                )
            else:
                distance_profiles[i] = _squared_distance_profile(
                    QT_all[i],
                    self.X_ssqs_[i],
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
def _squared_distance_profile(QT, T_ssq, Q):
    """
    Compute squared Euclidean distance profile between query and a time series.

    This function calculates the squared distance profile for a single time series by
    leveraging the dot product of the query and time series as well as precomputed sums
    of squares to efficiently compute the squared distances.

    Parameters
    ----------
    QT : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The dot product between the query and the time series.
    T_ssq : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The sliding sum-of-squares of the time series, i.e. for each channel and
        candidate position the sum of squared values over the ``query_length`` window
        starting at that position. Precomputed at fit time.
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
    distance_profile = np.zeros(profile_length)
    for k in range(n_channels):
        _qsum = 0
        for j in range(query_length):
            _qsum += Q[k, j] ** 2

        # Fold the -2*QT scaling and the query sum-of-squares into the loop to
        # avoid allocating profile-length temporaries per channel.
        for i in range(profile_length):
            distance_profile[i] += _qsum + T_ssq[k, i] - 2 * QT[k, i]
    return distance_profile


def _sliding_sum_of_squares(X, query_length):
    """
    Compute the sliding window sum-of-squares of every series in a collection.

    Parameters
    ----------
    X : np.ndarray, shape=(n_cases, n_channels, n_timepoints)
        Collection of equal-length time series.
    query_length : int
        The window length over which squared values are summed.

    Returns
    -------
    ssq : np.ndarray, shape=(n_cases, n_channels, n_timepoints - query_length + 1)
        For each case, channel and candidate position, the sum of squared values
        over the ``query_length`` window starting at that position.
    """
    sq = X.astype(np.float64) ** 2
    # Prefix sums along time; sliding window sum = cumsum difference.
    csum = np.cumsum(sq, axis=2)
    n_timepoints = X.shape[2]
    profile_length = n_timepoints - query_length + 1
    ssq = np.empty((X.shape[0], X.shape[1], profile_length))
    ssq[:, :, 0] = csum[:, :, query_length - 1]
    if profile_length > 1:
        ssq[:, :, 1:] = csum[:, :, query_length:] - csum[:, :, : profile_length - 1]
    return ssq


def _batched_sliding_dot_product(q, X_spectra, fft_len, n_timepoints):
    """
    Compute the sliding dot product of a query against many series via FFT.

    The reversed query is transformed once per channel and broadcast-multiplied
    against the cached series spectra, then a single batched inverse FFT yields
    the sliding dot products (correlations) for every case.

    Parameters
    ----------
    q : np.ndarray, shape=(n_channels, query_length)
        The query subsequence.
    X_spectra : np.ndarray, shape=(n_cases, n_channels, fft_len // 2 + 1)
        The cached ``rfft`` of every fitted series (computed at fit time).
    fft_len : int
        The FFT length used to compute ``X_spectra``.
    n_timepoints : int
        The number of timepoints of the fitted series.

    Returns
    -------
    QT : np.ndarray, shape=(n_cases, n_channels, n_timepoints - query_length + 1)
        The sliding dot product between the query and each fitted series.
    """
    query_length = q.shape[1]
    rev_q = q[:, ::-1]
    Q_spectrum = scipy.fft.rfft(rev_q, n=fft_len, axis=1)
    prod = X_spectra * Q_spectrum[np.newaxis, :, :]
    conv = scipy.fft.irfft(prod, n=fft_len, axis=2)
    # "valid" slice of the full linear convolution.
    return conv[:, :, query_length - 1 : n_timepoints]


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
        for k in range(n_channels):
            sub_is_constant = T_stds[k, i] <= AEON_NUMBA_STD_THRESHOLD
            # Two Constant case
            if Q_is_constant[k] and sub_is_constant:
                _val = 0
            # One Constant case
            elif Q_is_constant[k] or sub_is_constant:
                _val = query_length
            else:
                denom = query_length * Q_stds[k] * T_stds[k, i]

                p = (QT[k, i] - query_length * (Q_means[k] * T_means[k, i])) / denom
                p = min(p, 1.0)

                _val = abs(2 * query_length * (1.0 - p))
            distance_profile[i] += _val

    return distance_profile
