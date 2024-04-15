"""Matrix Profile Distances."""

import numpy as np
from numba import njit


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
        d[i] = (
            2
            * q_len
            * (
                1
                - (
                    (dot_prod[i] - q_len * q_mean * t_mean[i])
                    / (q_len * q_std * t_std[i])
                )
            )
        )

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
    if m == 0:
        if len_x > len_y:
            m = int(len_x / 4)
        else:
            m = int(len_y / 4)

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


def mpdist(x: np.ndarray, y: np.ndarray, m: int = 0) -> float:
    """Matrix Profile Distance.

    Parameters
    ----------
    x : np.ndarray
        First time series, univariate, shape ``(n_timepoints,)``
    y : np.ndarray
        Second time series, univariate, shape ``(n_timepoints,)``
    m : int (default = 0)
        Length of the subsequence

    Returns
    -------
    float
        Matrix Profile distance between x and y

    Raises
    ------
    ValueError
        If x and y are not 1D arrays

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_distance
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> mpdist(x, y)
    31.622776601683793
    """
    if x.ndim == 1 and y.ndim == 1:
        return _mpdist(x, y, m)
    raise ValueError("x and y must be a 1D array of shape (n_timepoints,)")


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
