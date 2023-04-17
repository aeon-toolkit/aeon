# -*- coding: utf-8 -*-
"""General numba utilities for dilated shapelet transform."""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def prime_up_to(n):
    """Check if any number from one to n is a prime number the ones who are.

    Parameters
    ----------
    n : int
        Number up to which the search for prime number will go.

    Returns
    -------
    array
        Prime numbers up to n.
    """
    is_p = np.zeros(n + 1, dtype=np.bool_)
    for i in range(n + 1):
        is_p[i] = is_prime(i)
    return np.where(is_p)[0]


@njit(cache=True, fastmath=True)
def is_prime(n):
    """Check if the passed number is a prime number.

    Parameters
    ----------
    n : int
        A number to test

    Returns
    -------
    bool
        Wheter n is a prime number

    """
    if (n % 2 == 0 and n > 2) or n == 0 or n == 1:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if not n % i:
            return False
    return True


@njit(cache=True, fastmath=True)
def choice_log(n_choice, n_sample):
    """Random choice function with log probability rather than uniform.

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
        The randomly choosen samples.

    """
    if n_choice > 1:
        # Define log probas for each choices
        P = np.array([1 / 2 ** np.log(i) for i in range(1, n_choice + 1)])
        # Bring everything between 0 and 1 as a cumulative probability
        P = P.cumsum() / P.sum()
        loc = np.zeros(n_sample, dtype=np.int64)
        for i in prange(n_sample):
            loc[i] = np.where(P >= np.random.rand())[0][0]
        return loc
    else:
        return np.zeros(n_sample, dtype=np.int64)


@njit(cache=True, fastmath=True)
def get_subsequence(X, i_start, length, dilation):
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


@njit(cache=True, fastmath=True)
def get_subsequence_with_mean_std(X, i_start, length, dilation):
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
    values = np.zeros((n_channels, length))
    means = np.zeros(n_channels)
    stds = np.zeros(n_channels)

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
        stds[i_channel] = ((_sum2 / length) - means[i_channel] ** 2) ** 0.5

    return values, means, stds


@njit(cache=True, fastmath=True)
def sliding_mean_std_one_series(X, length, dilation):
    """Return the mean and standard deviation for all subsequence (l,d) in X.

    Parameters
    ----------
    X : array, shape (n_channels, n_timestamps)
        An input time series
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet

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
    mean = np.zeros((n_channels, n_subs))
    std = np.zeros((n_channels, n_subs))

    for i_channel in prange(n_channels):
        mod_dil = n_subs % dilation
        for i_mod_dilation in prange(dilation):
            _idx = i_mod_dilation
            _sum = 0
            _sum2 = 0
            # Init First sums
            for _ in prange(length):
                _v = X[i_channel, _idx]
                _sum += _v
                _sum2 += _v * _v
                _idx += dilation

            mean[i_channel, i_mod_dilation] = _sum / length
            _s = (_sum2 / length) - (mean[i_channel, i_mod_dilation] ** 2)
            if _s > 0:
                std[i_channel, i_mod_dilation] = _s**0.5
            # Number of remaining subsequence for each starting i_mod_dilation index

            n_subs_mod_d = n_subs // dilation
            if mod_dil > 0:
                n_subs_mod_d += 1
                mod_dil -= 1
            # Iteratively update sums
            start_idx_sub = i_mod_dilation + dilation
            for _ in prange(1, n_subs_mod_d):
                # New value, not present in the previous subsequence
                _v_new = X[i_channel, start_idx_sub + ((length - 1) * dilation)]
                # Index of the old value, not present in the current subsequence
                _v_old = X[i_channel, start_idx_sub - dilation]

                _sum += _v_new - _v_old
                _sum2 += (_v_new * _v_new) - (_v_old * _v_old)

                mean[i_channel, start_idx_sub] = _sum / length
                _s = (_sum2 / length) - (mean[i_channel, start_idx_sub] ** 2)
                if _s > 0:
                    std[i_channel, start_idx_sub] = _s**0.5
                start_idx_sub += dilation
    return mean, std


@njit(cache=True, fastmath=True)
def sliding_dot_product(X, values, length, dilation):
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
def combinations_1d(x, y):
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
    combinations = np.zeros((u_mask.sum(), 2), dtype=np.int64)
    i_comb = 0
    for i in range(x.shape[0]):
        if u_mask[np.where(u_x == x[i])[0][0], np.where(u_y == y[i])[0][0]]:
            combinations[i_comb, 0] = x[i]
            combinations[i_comb, 1] = y[i]
            u_mask[np.where(u_x == x[i])[0][0], np.where(u_y == y[i])[0][0]] = False
            i_comb += 1
    return combinations
