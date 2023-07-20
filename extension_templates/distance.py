# -*- coding: utf-8 -*-
"""
Extension template for time series distance function.

This is a quick guide on how to implement a new aeon distance function. A wish list
for distance functions we would like to see is here
https://github.com/aeon-toolkit/aeon/issues?q=is%3Aopen+is%3Aissue+label%3Adistances+

You need to implement three public functions for a distance called "foo"

TODO 1: foo_distance: distance between two series, returns a float

Optional:
TODO 2: foo_pairwise_distance: distance between a collection of series, returns a matrix
TODO 3: foo_alignment_path: path constructed to find distance, returns a list of (x,
y) pairs
TODO 4: foo_cost_matrix: path constructed to find distance, returns a list of (x,
y) pairs

Note many elastic distances calculate a cost
matrix. Look at any of the distances to see how we structure this, but ultimately it
is your choice how to internally design the calculationds. Please use numba where
ever possible.

To contribute the distance to aeon, you need to adjust the file
aeon/distance/_distance.py to make sure it is tested and available via the function
distance. There are three things to do

TODO 5: function distance
in the function
    def distance(
        x: np.ndarray,
        y: np.ndarray,
        metric: Union[str, DistanceFunction],
        **kwargs: Any,
    ) -> float:
add an if clause here returning a distance with relevant parameters (necessary because
numba cant handle kwargs)
    elif metric == "foo":
        return foo_distance(
            x,
            y,
            kwargs.get("para1"),
            kwargs.get("para2"),
        )
TODO 6: function pairwise_distance (if foo_pairwise_distance implemented)
in the function
    def pairwise_distance(
        x: np.ndarray,
        y: np.ndarray = None,
        metric: Union[str, DistanceFunction] = None,
        **kwargs: Any,
    ) -> np.ndarray:
ad an if clause returning the pairwise distance.
    elif metric == "foo":
        return foo_pairwise_distance(x, y, kwargs.get("para1", kwargs.get("para2"))
TODO 7: DISTANCES list
Add your distance to the list of DISTANCES used in testing
DISTANCES = [
    {
        "name": "foo",
        "distance": foo_distance,
        "pairwise_distance": foo_pairwise_distance,
        "cost_matrix": foo_cost_matrix,
        "alignment_path": foo_alignment_path,
    },
"""
import numpy as np
from numba import njit


# TODO 1: distance function
# Give function a sensible name
# The decorator means it will be JIT compiled and be much much faster!
# distance functions should accept both 1D and 2D arrrays (1D are univariate time
# series shape (n_timepoints), 2D are multivariate (n_channels, n_timepoints). The
# n_channels should match for both series, but distance functions should be able to
# handle unequal length series.
@njit(cache=True, fastmath=True)
def foo_distance(
    x: np.ndarray, y: np.ndarray, para1: float = 0.1, para2: int = 3
) -> float:
    r"""Compute the my_distance between two time series.

    # give docstring formal definition of foo distance. We use a meaningless
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
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> foo_distance(x, y, para1)
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
    raise ValueError("x and y must be 1D or 2D and of the same dinension")


# TODO 2: Implement pairwise distances
# Pairwise distance functions return a matrix of distances between sets of time
# series. They take two arguments: the first argument, X, is required and must be a
# collection. This is important, because it is used to infer the type of the optional
# second argument, y.
# If it is 3D (X.ndim == 3) then it is a collection (n_cases, n_channels,
# n_timepoints) assume it
# is
@njit(cache=True, fastmath=True)
def foo_pairwise_distance(
    X: np.ndarray, y: np.ndarray = None, para1: float = None, para2: float = None
) -> np.ndarray:
    """Compute the pairwise my distance between a collection of time series.

    For pairwise, 2D np.ndarray are treated as a collection of 1D series,
    and 3D np.ndarray as a collection of 2D series.

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
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> my_pairwise_distance(X)
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
