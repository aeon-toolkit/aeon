"""Matrix Profile Distances."""

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


def mp_distance(x: np.ndarray, y: np.ndarray, m: int = 0) -> float:
    r"""Matrix Profile Distance.

    MPdist [2]_ is a distance method based on the matrix profile [1]_. Given a
    window length $m$, the matrix profile between two series $x$ and $y$, denoted
    $P_{xy}$, is a new time series where each point $i$ stores the Euclidean distance
    between `x[i:i+m]` and the nearest neighbour window to x[i:i+m]` in $y$ . MPdist
    is found by concatenating $P_{xy}$ and $P_{yx}$, sorting the distances
    into ascending order then taking the $k^{th}$ smallest as the distance. $k$ is
    set to 5% of the sum of the lengths of the two time series.

    This function supports unequal length series. We recommend using MPDist with
    normalised series, otherwise the distance may be dominated by differences in scale.

    Parameters
    ----------
    x : np.ndarray
        First time series, univariate, shape ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, univariate, shape ``(n_timepoints,)``.
    m : int (default = 0)
        Length of the sliding window. If 0, it is set to 1/4 of the length of the
        shortest time series.

    Returns
    -------
    float
        Matrix Profile distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D arrays

    References
    ----------
    .. [1] S. Gharghabi, S. Imani, A. Bagnall, A. Darvishzadeh and E. Keogh,
    "Matrix Profile XII: MPdist: A Novel Time Series Distance Method to Allow
    Data Mining in More Challenging Scenarios", 2018 IEEE International Conference
    on Data Mining (ICDM), 2018.
    .. [2] S. Gharghabi, S. Imani, A. Bagnall, A. Darvishzadeh and E. Keogh
    "An ultra-fast time series distancemethod to allow data mining inmore complex
    real-world deployments", Data Mining and Knowledge Discovery, 34(5), 2020.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mp_distance
    >>> x = np.array([5, 9, 16, 23, 19, 13, 7])
    >>> y = np.array([3, 7, 13, 19, 23, 31, 36, 40, 48, 55, 63])
    >>> m = 4
    >>> mp_distance(x, y, m) # doctest: +SKIP
    0.05663764013361034
    """
    x = np.squeeze(x)
    y = np.squeeze(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be a 1D array of shape (n_timepoints,)")
    len_x = len(x)
    len_y = len(y)

    if m < 0:
        raise ValueError(
            "subseries length must be greater than 0 or zero to default "
            "to 1/4 of the length of the shortest time series"
        )
    elif m == 0:
        if len_x > len_y:
            m = int(len_x / 4)
        else:
            m = int(len_y / 4)
    if m > len_x or m > len_y:
        raise ValueError(
            "subseries length must be less than or equal to the length "
            "of both time series"
        )
    return _mpdist(x, y, m)


def _mpdist(x: np.ndarray, y: np.ndarray, m: int) -> float:
    threshold = 0.05
    len_x = len(x)
    len_y = len(y)

    first_dot_prod_ab = _sliding_dot_products(y[0:m], x, m, len_x)
    dot_prod_ab = _sliding_dot_products(x[0:m], y, m, len_y)
    mp_ab, ip_ab = _stomp_ab(
        x, y, m, first_dot_prod_ab, dot_prod_ab
    )  # AB Matrix profile

    first_dot_prod_ba = _sliding_dot_products(x[0:m], y, m, len_y)
    dot_prod_ba = _sliding_dot_products(y[0:m], x, m, len_x)
    mp_ba, ip_ba = _stomp_ab(
        y, x, m, first_dot_prod_ba, dot_prod_ba
    )  # BA Matrix profile

    join_mp = np.concatenate([mp_ab, mp_ba])

    k = int(np.ceil(threshold * (len(x) + len(y))))

    sorted_mp = np.sort(join_mp)

    if len(sorted_mp) > k:
        dist = sorted_mp[k]
    else:
        dist = sorted_mp[len(sorted_mp) - 1]

    return dist


def _sliding_dot_products(q, t, len_q, len_t):
    """
    Compute the sliding dot products between a query and a time series.

    Parameters
    ----------
        q: numpy.array
            Query.
        t: numpy.array
            Time series.
        len_q: int
            Length of the query.
        len_t: int
            Length of the time series.

    Output
    ------
        dot_prod: numpy.array
             Sliding dot products between q and t.
    """
    # Reversing query and padding both query and time series
    padded_t = np.pad(t, (0, len_t))
    reversed_q = np.flipud(q)
    padded_reversed_q = np.pad(reversed_q, (0, 2 * len_t - len_q))

    # Applying FFT to both query and time series
    fft_t = np.fft.fft(padded_t)
    fft_q = np.fft.fft(padded_reversed_q)

    # Applying inverse FFT to obtain the convolution of the time series by
    # the query
    element_wise_mult = np.multiply(fft_t, fft_q)
    inverse_fft = np.fft.ifft(element_wise_mult)

    # Returns only the valid dot products from inverse_fft
    dot_prod = inverse_fft[len_q - 1 : len_t].real

    return dot_prod


@njit(cache=True, fastmath=True)
def _calculate_distance_profile(
    dot_prod, q_mean, q_std, t_mean, t_std, q_len, n_t_subs
):
    """
    Calculate the distance profile for the given query.

    Parameters
    ----------
        dot_prod: numpy.array
            Sliding dot products between the time series and the query.
        q_mean: float
            Mean of the elements of the query.
        q_std: float
            Standard deviation of elements of the query.
        t_mean: numpy.array
            Array with the mean of the elements from each subsequence of
            length(query) from the time series.
        t_std: numpy.array
            Array with the standard deviation of the elements from each
            subsequence of length(query) from the time series.
        q_len: int
            Length of the query.
        n_t_subs: int
            Number of subsequences in the time series.

    Output
    ------
        d: numpy.array
            Distance profile of query q.
    """
    d = np.empty(n_t_subs)
    for i in range(n_t_subs):
        temp = (dot_prod[i] - q_len * q_mean * t_mean[i]) / (q_len * q_std * t_std[i])
        d[i] = 2 * q_len * (1 - temp)

    d = np.absolute(d)
    d = np.sqrt(d)

    return d


@njit(cache=True, fastmath=True)
def _stomp_ab(
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    first_dot_prod: np.ndarray,
    dot_prod: np.ndarray,
):
    """
    STOMP implementation for AB similarity join.

    Parameters
    ----------
        x: numpy.array
            First time series.
        y: numpy.array
            Second time series.
        m: int
            Length of the subsequences.
        first_dot_prod: np.ndarray
            The distance profile for the first y subsequence.
        dot_prod: np.ndarray
            the distance profile for the first x subsequence.

    Output
    ------
        mp: numpy.array
            Array with the distance between every subsequence from x
            to the nearest subsequence with same length from y.
        ip: numpy.array
            Array with the index of the nearest neighbor of x in y.
    """
    len_x = len(x)
    len_y = len(y)

    # Number of subsequences
    subs_x = len_x - m + 1
    subs_y = len_y - m + 1

    # Compute the mean and standard deviation
    x_mean = []
    x_std = []
    y_mean = []
    y_std = []

    for i in range(subs_x):
        x_mean.append(np.mean(x[i : i + m]))
        x_std.append(np.std(x[i : i + m]))

    for i in range(subs_y):
        y_mean.append(np.mean(y[i : i + m]))
        y_std.append(np.std(y[i : i + m]))

    # Initialization
    mp = np.full(subs_x, np.inf)  # matrix profile
    ip = np.zeros(subs_x)  # index profile

    dp = _calculate_distance_profile(
        dot_prod, x_mean[0], x_std[0], y_mean, y_std, m, subs_y
    )

    # Update the matrix profile
    mp[0] = np.amin(dp)
    ip[0] = np.argmin(dp)

    for i in range(1, subs_x):
        for j in range(subs_y - 1, 0, -1):
            dot_prod[j] = (
                dot_prod[j - 1] - y[j - 1] * x[i - 1] + y[j - 1 + m] * x[i - 1 + m]
            )
        # Compute the next dot products using previous ones
        dot_prod[0] = first_dot_prod[i]
        dp = _calculate_distance_profile(
            dot_prod, x_mean[i], x_std[i], y_mean, y_std, m, subs_y
        )
        mp[i] = np.amin(dp)
        ip[i] = np.argmin(dp)

    return mp, ip


def mp_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    m: int = 0,
) -> np.ndarray:
    """Compute the mpdist pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)``.
        If None, then the mpdist pairwise distance between the instances of X is
        calculated.
    m : int (default = 0)
        Length of the subsequence

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        mpdist pairwise matrix between the instances of X if only X is given
        else mpdist pairwise matrix between the instances of X and y.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mp_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[16, 23, 19, 13],[48, 55, 63, 67]])
    >>> mp_pairwise_distance(X, m = 3)
    array([[0.        , 1.56786235],
           [1.56786235, 0.        ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[21, 13, 9]],[[19, 14, 5]], [[17, 11, 6]]])
    >>> mp_pairwise_distance(X, y, m = 2)
    array([[2.82842712, 2.82842712, 2.82842712],
           [2.82842712, 2.82842712, 2.82842712],
           [2.82842712, 2.82842712, 2.82842712]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[22, 18, 12]])
    >>> mp_pairwise_distance(X, y_univariate, m = 2)
    array([[2.82842712],
           [2.82842712],
           [2.82842712]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if m == 0:
        m = int(_X.shape[2] / 4)

    if y is None:
        return _mpdist_pairwise_distance_single(_X, m)

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )

    return _mpdist_pairwise_distance(_X, _y, m)


def _mpdist_pairwise_distance_single(x: NumbaList[np.ndarray], m: int) -> np.ndarray:
    n_cases = len(x)
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = mp_distance(x[i], x[j], m)
            distances[j, i] = distances[i, j]

    return distances


def _mpdist_pairwise_distance(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray], m: int
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)

    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = mp_distance(x[i], y[j], m)
    return distances
