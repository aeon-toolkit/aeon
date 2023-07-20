# -*- coding: utf-8 -*-
"""
Extension template for time series distance function.

This is a quick guide on how to implement a new aeon distance function. A wish list
for distance functions we would like to see is here
https://github.com/aeon-toolkit/aeon/issues?q=is%3Aopen+is%3Aissue+label%3Adistances+

You need to implement three public functions for a distance called "my"

my_distance: distance between two series, returns a float
my_pairwise_distance: distance between a collection of series, returns a matrix
my_alignment_path: path constructed to find distance, returns a list of (x,y) pairs

Note many elastic distances calculate a cost
matrix. Look at any of the distances to see how we structure this, but ultimately it
is your choice how to internally design the calculationds. Please use numba where
ever possible.
"""
import numpy as np
from numba import njit


# The decorator means it will be JIT compiled and be much much faster!
# TODO: give function a sensible name
@njit(cache=True, fastmath=True)
def my_distance(
    x: np.ndarray, y: np.ndarray, para1: float = 0.1, para2: int = 3
) -> float:
    r"""Compute the my_distance between two time series.

    TODO given docstring formal definition of my distance. We use a meaningless
    definition as an example

    .. math::
        my_distance(x, y) = para1*(x[0] - y[0])+para2

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    para1 : float, default = 0.1
        Parameter for distance, usually a float or int.
    para2 : int, default = 3
        Parameter for distance, usually a float or int.

    Returns
    -------
    float
        My distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    TODO give example output
    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> my_distance(x, y, para1)
    -11
    """
    # Note our distances are configured to work with either 1D or 2D
    # TODO implement logic for 1D series
    if x.ndim == 1 and y.ndim == 1:
        return para1 * (x[0] - y[0]) + para2

    # TODO implement logic for 2D series
    if x.ndim == 2 and y.ndim == 2:
        sum = 0
        for i in range(x.shape[0]):
            sum = sum + x[i][0] - y[i][0]
        return para1 * sum + para2
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def my_pairwise_distance(
    X: np.ndarray, y: np.ndarray = None, para1: float = None, para2: float = None
) -> np.ndarray:
    """Compute the pairwise my distance between a collection of time series.

    For pairwise, 2D np.ndarray are treated as a collection of 1D series,
    and 3D np.ndarray as a collection of 2D series
    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,), default=None
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        dtw pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> dtw_pairwise_distance(X)
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_pairwise_distance(X, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    """
    return 0
