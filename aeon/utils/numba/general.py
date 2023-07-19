# -*- coding: utf-8 -*-
"""General numba utilities."""

__author__ = ["MatthewMiddlehurst", "baraline"]
__all__ = [
    "unique_count",
    "first_order_differences",
    "first_order_differences_2d",
    "first_order_differences_3d",
    "z_normalise_series_with_mean",
    "z_normalise_series",
    "z_normalise_series_2d",
    "z_normalise_series_3d",
    "set_numba_random_seed",
    "choice_log",
    "get_subsequence",
    "get_subsequence_with_mean_std",
    "sliding_mean_std_one_series",
    "sliding_dot_product",
    "combinations_1d",
]

from typing import Tuple

import numpy as np
from numba import njit, prange

import aeon.utils.numba.stats as stats


@njit(fastmath=True, cache=True)
def unique_count(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numba unique value count function for a 1d numpy array.

    np.unique() is supported by numba, but the return_counts parameter is not.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    unique : 1d numpy array
        The unique values in X
    counts : 1d numpy array
        The occurrence count for each unique value in X

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import unique_count
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> unique, counts = unique_count(X)
    """
    if X.shape[0] > 0:
        X = np.sort(X)
        unique = np.zeros(X.shape[0])
        unique[0] = X[0]
        counts = np.zeros(X.shape[0], dtype=np.int32)
        counts[0] = 1
        uc = 0

        for i in X[1:]:
            if i != unique[uc]:
                uc += 1
                unique[uc] = i
                counts[uc] = 1
            else:
                counts[uc] += 1
        return unique[: uc + 1], counts[: uc + 1]
    return np.zeros(0), np.zeros(0, dtype=np.int32)


@njit(fastmath=True, cache=True)
def first_order_differences(X: np.ndarray) -> np.ndarray:
    """Numba first order differences function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    arr : 1d numpy array of size (X.shape[0] - 1)
        The first order differences of X

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import first_order_differences
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> diff = first_order_differences(X)
    """
    return X[1:] - X[:-1]


@njit(fastmath=True, cache=True)
def first_order_differences_2d(X: np.ndarray) -> np.ndarray:
    """Numba first order differences function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 2d numpy array of shape (X.shape[0], X.shape[1] - 1)
        The first order differences for axis 1 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import first_order_differences_2d
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> diff = first_order_differences_2d(X)
    """
    return X[:, 1:] - X[:, :-1]


@njit(fastmath=True, cache=True)
def first_order_differences_3d(X: np.ndarray) -> np.ndarray:
    """Numba first order differences function for a 3d numpy array.

    Parameters
    ----------
    X : 3d numpy array
        A 3d numpy array of values

    Returns
    -------
    arr : 2d numpy array of shape (X.shape[0], X.shape[1], X.shape[2] - 1)
        The first order differences for axis 2 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import first_order_differences_3d
    >>> X = np.array([[[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]]])
    >>> diff = first_order_differences_3d(X)
    """
    return X[:, :, 1:] - X[:, :, :-1]


@njit(fastmath=True, cache=True)
def z_normalise_series_with_mean(X: np.ndarray, series_mean: float) -> np.ndarray:
    """Numba series normalization function for a 1d numpy array with mean.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values
    series_mean : float
        The mean of the series

    Returns
    -------
    arr : 1d numpy array
        The normalised series

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import z_normalise_series_with_mean
    >>> from aeon.utils.numba.stats import mean
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> X_norm = z_normalise_series_with_mean(X, mean(X))
    """
    s = stats.std(X)
    if s > 0:
        arr = (X - series_mean) / s
    else:
        arr = X - series_mean
    return arr


@njit(fastmath=True, cache=True)
def z_normalise_series(X: np.ndarray) -> np.ndarray:
    """Numba series normalization function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The normalised series

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import z_normalise_series
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> X_norm = z_normalise_series(X)
    """
    s = stats.std(X)
    if s > 0:
        arr = (X - stats.mean(X)) / s
    else:
        arr = X - stats.mean(X)
    return arr


@njit(fastmath=True, cache=True)
def z_normalise_series_2d(X: np.ndarray) -> np.ndarray:
    """Numba series normalization function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 2d numpy array
        The normalised series

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import z_normalise_series_2d
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> X_norm = z_normalise_series_2d(X)
    """
    arr = np.zeros(X.shape)
    for i in range(X.shape[0]):
        arr[i] = z_normalise_series(X[i])
    return arr


@njit(fastmath=True, cache=True)
def z_normalise_series_3d(X: np.ndarray) -> np.ndarray:
    """Numba series normalization function for a 3d numpy array.

    Parameters
    ----------
    X : 3d numpy array
        A 3d numpy array of values

    Returns
    -------
    arr : 3d numpy array
        The normalised series

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.numba.general import z_normalise_series_3d
    >>> X = np.array([
    ...     [[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]],
    ...     [[4, 4, 4, 4, 3, 3, 3, 2, 2, 1], [8, 8, 8, 8, 7, 7, 7, 6, 6, 5]],
    ... ])
    >>> X_norm = z_normalise_series_3d(X)
    """
    arr = np.zeros(X.shape)
    for i in range(X.shape[0]):
        arr[i] = z_normalise_series_2d(X[i])
    return arr


@njit()
def set_numba_random_seed(seed: int) -> None:
    """Set the random seed for numba.

    Parameters
    ----------
    seed : int
        The seed to set.
    """
    if seed is not None:
        np.random.seed(seed)


@njit(fastmath=True, cache=True)
def choice_log(n_choice: int, n_sample: int) -> np.ndarray:
    """Random choice function with log probability rather than uniform.

    To seed the function the `np.random.seed` must be set in a numba function prior to
    calling this i.e. using `set_numba_random_seed`.

    Parameters
    ----------
    n_choice : int
        The number of possible choice. Choices will be made in an array from 0 to
        n_choice-1.
    n_sample : int
        Number of choice to sample.

    Returns
    -------
    array
        The randomly chosen samples.
    """
    if n_choice > 1:
        # Define log probas for each choice
        P = np.array([1 / 2 ** np.log(i) for i in range(1, n_choice + 1)])
        # Bring everything between 0 and 1 as a cumulative probability
        P = P.cumsum() / P.sum()
        loc = np.zeros(n_sample, dtype=np.int32)
        for i in prange(n_sample):
            loc[i] = np.where(P >= np.random.rand())[0][0]
        return loc
    else:
        return np.zeros(n_sample, dtype=np.int32)


@njit(fastmath=True, cache=True)
def get_subsequence(
    X: np.ndarray, i_start: int, length: int, dilation: int
) -> np.ndarray:
    """Get a subsequence from a time series given a starting index.

    Parameters
    ----------
    X : array, shape (n_channels, n_timestamps)
        Input time series.
    i_start : int
        A starting index between [0, n_timestamps - (length-1)*dilation]
    length : int
        Length parameter of the subsequence.
    dilation : int
        Dilation parameter of the subsequence.

    Returns
    -------
    values : array, shape (length)
        The resulting subsequence.
    """
    n_channels, _ = X.shape
    values = np.zeros((n_channels, length))
    idx = i_start
    for i_length in prange(length):
        values[:, i_length] = X[:, idx]
        idx += dilation

    return values


@njit(fastmath=True, cache=True)
def get_subsequence_with_mean_std(
    X: np.ndarray, i_start: int, length: int, dilation: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get a subsequence, its mean and std from a time series given a starting index.

    Parameters
    ----------
    X : array, shape (n_channels, n_timestamps)
        Input time series.
    i_start : int
        A starting index between [0, n_timestamps - (length-1)*dilation]
    length : int
        Length parameter of the subsequence.
    dilation : int
        Dilation parameter of the subsequence.

    Returns
    -------
    values : array, shape (n_channels, length)
        The resulting subsequence.
    means : array, shape (n_channels)
        The mean of each channel
    stds : array, shape (n_channels)
        The std of each channel
    """
    n_channels, _ = X.shape
    values = np.zeros((n_channels, length), dtype=np.float64)
    means = np.zeros(n_channels, dtype=np.float64)
    stds = np.zeros(n_channels, dtype=np.float64)
    for i_channel in prange(n_channels):
        _sum = 0
        _sum2 = 0
        idx = i_start
        for i_length in prange(length):
            _v = X[i_channel, idx]

            _sum += _v
            _sum2 += _v * _v

            values[i_channel, i_length] = _v
            idx += dilation

        means[i_channel] = _sum / length
        _s = (_sum2 / length) - (means[i_channel] ** 2)
        if _s > 0:
            stds[i_channel] = _s**0.5

    return values, means, stds


@njit(fastmath=True, cache=True)
def sliding_mean_std_one_series(
    X: np.ndarray, length: int, dilation: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation for all subsequence (l,d) in X.

    Parameters
    ----------
    X : array, shape (n_channels, n_timestamps)
        An input time series
    length : int
        Length of the subsequence
    dilation : int
        Dilation of the subsequence

    Returns
    -------
    mean : array, shape (n_channels, n_timestamps - (length-1) * dilation)
        The mean of each subsequence with parameter length and dilation in X.
    std : array, shape (n_channels, n_timestamps - (length-1) * dilation)
        The standard deviation of each subsequence with parameter length and dilation
        in X.
    """
    n_channels, n_timestamps = X.shape
    n_subs = n_timestamps - (length - 1) * dilation
    if n_subs <= 0:
        raise ValueError(
            "Invalid input parameter for sliding mean and std computations"
        )
    mean = np.zeros((n_channels, n_subs))
    std = np.zeros((n_channels, n_subs))

    for i_mod_dil in prange(dilation):
        # Array mainting indexes of a dilated subsequence
        _idx_sub = np.zeros(length, dtype=np.int32)
        for i_length in prange(length):
            _idx_sub[i_length] = (i_length * dilation) + i_mod_dil

        _sum = np.zeros(n_channels)
        _sum2 = np.zeros(n_channels)

        # Initialize first subsequence if it is valid
        if np.all(_idx_sub < n_timestamps):
            for i_length in prange(length):
                _idx_sub[i_length] = (i_length * dilation) + i_mod_dil
                for i_channel in prange(n_channels):
                    _v = X[i_channel, _idx_sub[i_length]]
                    _sum[i_channel] += _v
                    _sum2[i_channel] += _v * _v

            # Compute means and stds
            for i_channel in prange(n_channels):
                mean[i_channel, i_mod_dil] = _sum[i_channel] / length
                _s = (_sum2[i_channel] / length) - (mean[i_channel, i_mod_dil] ** 2)
                if _s > 0:
                    std[i_channel, i_mod_dil] = _s**0.5

        _idx_sub += dilation
        # As long as subsequences further subsequences are valid
        while np.all(_idx_sub < n_timestamps):
            # Update sums and mean stds arrays
            for i_channel in prange(n_channels):
                _v_new = X[i_channel, _idx_sub[-1]]
                _v_old = X[i_channel, _idx_sub[0] - dilation]
                _sum[i_channel] += _v_new - _v_old
                _sum2[i_channel] += (_v_new * _v_new) - (_v_old * _v_old)

                mean[i_channel, _idx_sub[0]] = _sum[i_channel] / length
                _s = (_sum2[i_channel] / length) - (mean[i_channel, _idx_sub[0]] ** 2)
                if _s > 0:
                    std[i_channel, _idx_sub[0]] = _s**0.5
            _idx_sub += dilation

    return mean, std


@njit(fastmath=True, cache=True)
def sliding_dot_product(
    X: np.ndarray, values: np.ndarray, length: int, dilation: int
) -> np.ndarray:
    """Compute a sliding dot product between a time series and a shapelet.

    Parameters
    ----------
    X : array, shape (n_channels, n_timestamps)
        An input time series
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet

    Returns
    -------
    dot_prods : array, shape (n_channels, n_timestamps - (length-1) * dilation)
        The dot products between each subsequence (l,d) of X and the value of the
        shapelet.
    """
    n_channels, n_timestamps = X.shape
    n_subs = n_timestamps - (length - 1) * dilation
    dot_prods = np.zeros((n_channels, n_subs))
    for i_sub in prange(n_subs):
        idx = i_sub
        for i_l in prange(length):
            for i_channel in prange(n_channels):
                dot_prods[i_channel, i_sub] += (
                    X[i_channel, idx] * values[i_channel, i_l]
                )
            idx += dilation
    return dot_prods


@njit(cache=True)
def combinations_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the unique pairs of the 2D array made by concatenating x and y.

    Parameters
    ----------
    x : array, shape=(u)
        A 1D array of values
    y : array, shape=(v)
        A 1D array of values

    Returns
    -------
    array, shape=(w, 2)
        The unique pairs in the concatenation of x and y.

    """
    # Fix issues with multiple length, but could be optimized
    u_x = np.unique(x)
    u_y = np.unique(y)
    u_mask = np.zeros((u_x.shape[0], u_y.shape[0]), dtype=np.bool_)

    for i in range(x.shape[0]):
        u_mask[np.where(u_x == x[i])[0][0], np.where(u_y == y[i])[0][0]] = True
    combinations = np.zeros((u_mask.sum(), 2), dtype=np.int32)
    i_comb = 0
    for i in range(x.shape[0]):
        if u_mask[np.where(u_x == x[i])[0][0], np.where(u_y == y[i])[0][0]]:
            combinations[i_comb, 0] = x[i]
            combinations[i_comb, 1] = y[i]
            u_mask[np.where(u_x == x[i])[0][0], np.where(u_y == y[i])[0][0]] = False
            i_comb += 1
    return combinations
