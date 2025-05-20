"""Dynamic time warping kernel (KDTW) distance between two time series."""

__maintainer__ = ["SebastianSchmidl"]

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.pointwise import squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def kdtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 0.125,
    epsilon: float = 1e-20,
    normalize: bool = True,
) -> float:
    r"""Compute the DTW kernel (KDTW) between two time series as a distance.

    KDTW is a similarity measure constructed from DTW and was introduced in [1]_. It has
    the property that it is invariant to shifts in the time series. The kernel is
    positive definite. This implementation provides a normalized distance [2]_ and
    takes the default values from [2]_. Details can be found online:
    https://people.irisa.fr/Pierre-Francois.Marteau/REDK/KDTW/KDTW.html

    Intuition of constructing a DTW kernel from DTW:
    Instead of keeping only one of the best alignment paths, the new kernel will try to
    sum up the costs of all the existing sub-sequence alignment paths with some
    weighting factor that will favor good alignments while penalizing bad alignments.

    The current implementation performs no bounding on the dynamic programming matrix
    (cost matrix) (uses no corridors).

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    gamma : float, default=0.125
        bandwidth parameter which weights the local contributions, i.e. the distances
        between locally aligned positions.
    epsilon : float, default=1e-20
        Small value to avoid zero. The default is 1e-20.
    normalize : bool, default=True
        Whether to normalize hte distance to make it shift invariant. The default is
        True.

    Returns
    -------
    float
        KDTW distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Pierre-François Marteau and Sylvie Gibet: On recursive edit distance kernels
    with application to time series classification. IEEE Transactions on Neural
    Networks and Learning Systems 26(6), 2014, pages 1121 - 1133.

    .. [2] Paparrizos, John, Chunwei Liu, Aaron J. Elmore, and Michael J. Franklin:
    Debunking Four Long-Standing Misconceptions of Time-Series Distance Measures. In
    Proceedings of the International Conference on Management of Data (SIGMOD),
    1887-1905, 2020. https://doi.org/10.1145/3318464.3389760.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import kdtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dist = kdtw_distance(x, y)
    4.613613743977605e-203
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return _kdtw_distance(_x, _y, gamma, epsilon, normalize)
    if x.ndim == 2 and y.ndim == 2:
        return _kdtw_distance(x, y, gamma, epsilon, normalize)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def kdtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 0.125,
    epsilon: float = 1e-20,
) -> np.ndarray:
    """Compute the cost matrix for KDTW between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    gamma : float, default=0.125
        bandwidth parameter which weights the local contributions, i.e. the distances
        between locally aligned positions.
    epsilon : float, default=1e-20
        Small value to avoid zero. The default is 1e-20.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        KDTW cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import kdtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> kdtw_cost_matrix(x, y)
    array([[ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.],
           [ 2.,  0.,  2.,  4.,  6.,  8., 10., 12., 14., 16.],
           [ 4.,  2.,  0.,  2.,  4.,  6.,  8., 10., 12., 14.],
           [ 6.,  4.,  2.,  0.,  2.,  4.,  6.,  8., 10., 12.],
           [ 8.,  6.,  4.,  2.,  0.,  2.,  4.,  6.,  8., 10.],
           [10.,  8.,  6.,  4.,  2.,  0.,  2.,  4.,  6.,  8.],
           [12., 10.,  8.,  6.,  4.,  2.,  0.,  2.,  4.,  6.],
           [14., 12., 10.,  8.,  6.,  4.,  2.,  0.,  2.,  4.],
           [16., 14., 12., 10.,  8.,  6.,  4.,  2.,  0.,  2.],
           [18., 16., 14., 12., 10.,  8.,  6.,  4.,  2.,  0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return _kdtw_cost_matrix(_x, _y, gamma, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        return _kdtw_cost_matrix(x, y, gamma, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _kdtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    epsilon: float,
    normalize: bool,
) -> float:
    # Do not subtract one because the cost matrix is one dimension larger:
    n = x.shape[-1]
    m = y.shape[-1]
    if normalize:
        self_x = _kdtw_cost_matrix(x, x, gamma, epsilon)[n, n]
        self_y = _kdtw_cost_matrix(y, y, gamma, epsilon)[m, m]
        norm_factor = np.sqrt(self_x * self_y)
    else:
        norm_factor = 1.0
    return _kdtw_cost_matrix(x, y, gamma, epsilon)[n, m] / norm_factor


@njit(cache=True, fastmath=True)
def _local_kernel(
    x: np.ndarray, y: np.ndarray, gamma: float, epsilon: float
) -> np.ndarray:
    factor = 1.0 / 3.0
    # Incoming shape is (n_channels, n_timepoints)
    # We want to calculate the multivariate squared distance between the two time series
    # considering each point in the time series as a separate instance, thus we need to
    # reshape to (m_cases, m_channels, 1), where m_cases = n_timepoints and
    # m_channels = n_channels.
    x = x.T.reshape(-1, x.shape[0], 1)
    y = y.T.reshape(-1, y.shape[0], 1)
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            # expects each input to have shape (n_channels, n_timepoints = 1)
            distances[i, j] = squared_distance(x[i], y[j])

    return factor * (np.exp(-distances / gamma) + epsilon)


@njit(cache=True, fastmath=True)
def _kdtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, gamma: float, epsilon: float
) -> np.ndarray:
    # deals with multivariate time series, afterward, we just work with the distances
    # and do not need to deal with the channels anymore
    local_kernel = _local_kernel(x, y, gamma, epsilon)

    # For the initial values of the cost matrix, we add 1
    n = np.shape(x)[-1] + 1
    m = np.shape(y)[-1] + 1

    cost_matrix = np.zeros((n, m))
    cumulative_dp_diag = np.zeros((n, m))
    diagonal_weights = np.zeros(max(n, m))

    # Initialize the diagonal weights
    min_timepoints = min(n, m)
    diagonal_weights[0] = 1.0
    for i in range(1, min_timepoints):
        diagonal_weights[i] = local_kernel[i - 1, i - 1]

    # Initialize the cost matrix and cumulative dp diagonal
    cost_matrix[0, 0] = 1
    cumulative_dp_diag[0, 0] = 1

    # - left column
    for i in range(1, n):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] * local_kernel[i - 1, 0]
        cumulative_dp_diag[i, 0] = cumulative_dp_diag[i - 1, 0] * diagonal_weights[i]

    # - top row
    for j in range(1, m):
        cost_matrix[0, j] = cost_matrix[0, j - 1] * local_kernel[0, j - 1]
        cumulative_dp_diag[0, j] = cumulative_dp_diag[0, j - 1] * diagonal_weights[j]

    # Perform the main dynamic programming loop
    for i in range(1, n):
        for j in range(1, m):
            local_cost = local_kernel[i - 1, j - 1]
            cost_matrix[i, j] = (
                cost_matrix[i - 1, j]
                + cost_matrix[i, j - 1]
                + cost_matrix[i - 1, j - 1]
            ) * local_cost
            cumulative_dp_diag[i, j] = (
                cumulative_dp_diag[i - 1, j] * diagonal_weights[i]
                + cumulative_dp_diag[i, j - 1] * diagonal_weights[j]
            )
            if i == j:
                cumulative_dp_diag[i, j] += (
                    cumulative_dp_diag[i - 1, j - 1] * local_cost
                )

    # Add the cumulative dp diagonal to the cost matrix
    cost_matrix = cost_matrix + cumulative_dp_diag
    return cost_matrix


def kdtw_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    gamma: float = 0.125,
    epsilon: float = 1e-20,
    normalize: bool = True,
) -> np.ndarray:
    """Compute the KDTW pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_instances, n_timepoints)``
        or ``(n_instances, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_instances, m_timepoints)`` or ``(m_instances, m_channels, m_timepoints)``.
        If None, then the KDTW pairwise distance between the instances of X is
        calculated.
    gamma : float, default=0.125
        bandwidth parameter which weights the local contributions, i.e. the distances
        between locally aligned positions.
    epsilon : float, default=1e-20
        Small value to avoid zero. The default is 1e-20.
    normalize : bool, default=True
        Whether to normalize the distance to make it shift invariant. The default is
        True.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        KDTW pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import kdtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> kdtw_pairwise_distance(X)
    array([[ 0.,  8., 12.],
           [ 8.,  0.,  8.],
           [12.,  8.,  0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> kdtw_pairwise_distance(X, y)
    array([[16., 19., 22.],
           [13., 16., 19.],
           [10., 13., 16.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> kdtw_pairwise_distance(X, y_univariate)
    array([[16.],
           [13.],
           [10.]])

    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)
    if y is None:
        # To self
        return _kdtw_pairwise_distance(_X, gamma, epsilon, normalize)

    _y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    return _kdtw_from_multiple_to_multiple_distance(_X, _y, gamma, epsilon, normalize)


@njit(cache=True, fastmath=True)
def _kdtw_pairwise_distance(
    X: NumbaList[np.ndarray], gamma: float, epsilon: float, normalize: bool
) -> np.ndarray:
    n_instances = len(X)
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _kdtw_distance(X[i], X[j], gamma, epsilon, normalize)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _kdtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    gamma: float,
    epsilon: float,
    normalize: bool,
) -> np.ndarray:
    n_instances = len(x)
    m_instances = len(y)
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _kdtw_distance(x[i], y[j], gamma, epsilon, normalize)
    return distances


@njit(cache=True)
def kdtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 0.125,
    epsilon: float = 1e-20,
    normalize: bool = True,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the kdtw alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    gamma : float, default=0.125
        bandwidth parameter which weights the local contributions, i.e. the distances
        between locally aligned positions.
    epsilon : float, default=1e-20
        Small value to avoid zero. The default is 1e-20.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The unnormalized kdtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import kdtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> kdtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 2.0)
    """
    x_size = x.shape[-1]
    y_size = y.shape[-1]
    cost_matrix = kdtw_cost_matrix(x, y, gamma, epsilon)
    return compute_min_return_path(cost_matrix), cost_matrix[x_size, y_size]
