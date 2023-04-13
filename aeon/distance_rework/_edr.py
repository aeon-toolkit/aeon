import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance
from aeon.distance_rework._bounding_matrix import create_bounding_matrix


@njit(cache=True, fastmath=True)
def edr_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> float:
    """Compute the edr distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    float
        edr distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import edr_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> edr_distance(x, y)
    0.0
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _edr_distance(x, y, bounding_matrix, epsilon)


@njit(cache=True, fastmath=True)
def edr_cost_matrix(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the edr cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray
        edr cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import edr_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> edr_cost_matrix(x, y)
    array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 0., 1., 2., 2., 2., 2., 2., 2., 2.],
           [1., 1., 0., 1., 2., 3., 3., 3., 3., 3.],
           [1., 2., 1., 0., 1., 2., 3., 4., 4., 4.],
           [1., 2., 2., 1., 0., 1., 2., 3., 4., 5.],
           [1., 2., 3., 2., 1., 0., 1., 2., 3., 4.],
           [1., 2., 3., 3., 2., 1., 0., 1., 2., 3.],
           [1., 2., 3., 4., 3., 2., 1., 0., 1., 2.],
           [1., 2., 3., 4., 4., 3., 2., 1., 0., 1.],
           [1., 2., 3., 4., 5., 4., 3., 2., 1., 0.]])
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _edr_cost_matrix(x, y, bounding_matrix, epsilon)


@njit(cache=True, fastmath=True)
def _edr_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float = None
) -> float:
    distance = _edr_cost_matrix(
        x, y, bounding_matrix, epsilon
    )[x.shape[1] - 1, y.shape[1] - 1]
    return float(distance / max(x.shape[1], y.shape[1]))


@njit(cache=True, fastmath=True)
def _edr_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float = None
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    if epsilon is None:
        epsilon = max(np.std(x), np.std(y)) / 4

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                squared_dist = univariate_squared_distance(x[:, i - 1], y[:, j - 1])
                if squared_dist < epsilon:
                    cost = 0
                else:
                    cost = 1
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + cost,
                    cost_matrix[i - 1, j] + 1,
                    cost_matrix[i, j - 1] + 1,
                )
    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def edr_pairwise_distance(
        X: np.ndarray, window: float = None, epsilon: float = None
) -> np.ndarray:
    """Compute the pairwise edr distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_dims, n_timepoints)
        The time series to compute the pairwise distance between.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray
        The pairwise edr distance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import edr_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> edr_pairwise_distance(X)
    array([[0.  , 0.75, 0.75],
           [0.75, 0.  , 0.75],
           [0.75, 0.75, 0.  ]])
    """
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _edr_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def edr_from_single_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
):
    """Compute the edr distance between a single time series and a set of time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        The time series to compute the distance from.
    y: np.ndarray (n_instances, n_dims, n_timepoints)
        The time series to compute the distance to.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray
        The edr distance between the single time series and the set of time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import edr_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 4]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> edr_from_single_to_multiple_distance(x, y)
    array([0.  , 0.75, 0.75])
    """
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _edr_distance(x, y[i], bounding_matrix, epsilon)

    return distances


@njit(cache=True, fastmath=True)
def edr_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = None
):
    """Compute the edr distance between a set of time series and a set of time series.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_dims, n_timepoints)
        The time series to compute the distance from.
    y: np.ndarray (n_instances, n_dims, n_timepoints)
        The time series to compute the distance to.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        The edr distance between two sets of time series, x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import edr_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> edr_from_multiple_to_multiple_distance(x, y)
    array([[0.75, 0.75, 1.  ],
           [1.  , 1.  , 1.  ],
           [1.  , 1.  , 0.75]])
    """
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _edr_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances
