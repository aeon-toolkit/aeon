from typing import Callable
import numpy as np

DistanceFunction: Callable[[np.ndarray, np.ndarray], float]
BoundingDistanceFunction: Callable[[np.ndarray, np.ndarray, np.ndarray], float]

def pairwise_distance(
        x: np.ndarray,
        y: np.ndarray,
        distance_function: DistanceFunction,
        *args,
        **kwargs
):
    """Compute pairwise distance between two sets of instances of time series.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_dims, n_timepoints)
        First set of instances of time series.
    y: np.ndarray (m_instances, n_dims, n_timepoints)
        Second set of instances of time series.
    distance_function
        Distance function that takes two time series as parameters. It should take
        the form distance_function(x: np.ndarray, y: np.ndarray).
    *args
        Additional positional arguments to pass to the distance function.
    **kwargs
        Additional keyword arguments to pass to the distance function.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        Pairwise distance matrix where [i, j] is the distance between the ith instance
        in x (x[i]) and the jth instance in y (y[j]).

    Example
    -------
    >>> from aeon.distances import euclidean_distance
    >>> x = np.array([[[1, 2, 3], [4, 5, 6]]])
    >>> y = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    >>> pairwise_distance(x, y, euclidean_distance)
    """
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    for i in range(n_instances):
        for j in range(m_instances):
            distances[i] = distance_function(x, y[i], *args, **kwargs)
            distances[j, i] = distances[i, j]

    return distances

def distance_from_single_to_multiple(
        x: np.ndarray,
        y: np.ndarray,
        distance_function: DistanceFunction,
        *args,
        **kwargs
):
    """Compute distance between a single time series and a set of instances of
    time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        A time series.
    y: np.ndarray (m_instances, n_dims, n_timepoints)
        A set of instances of time series.
    distance_function
        Distance function that takes two time series as parameters. It should take
        the form distance_function(x: np.ndarray, y: np.ndarray).
    *args
        Additional positional arguments to pass to the distance function.
    **kwargs
        Additional keyword arguments to pass to the distance function.


    Returns
    -------
    np.ndarray (n_instances)
        Array of distances where the ith instance in the result is the distance
        between y[i] and x.

    Example
    -------
    >>> from aeon.distances import euclidean_distance
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    >>> distance_from_single_to_multiple(x, y, euclidean_distance)
    """
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    for i in range(n_instances):
        distances[i] = distance_function(x, y[i], *args, **kwargs)
    return distances

def distance_from_multiple_to_multiple(
        x: np.ndarray,
        y: np.ndarray,
        distance_function: DistanceFunction,
        *args,
        **kwargs
):
    """Compute distance between one set of instances of time series and another set of
    instances of time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First set of instances of time series.
    y: np.ndarray (m_instances, n_dims, n_timepoints)
        Second set of instances of time series.
    distance_function
        Distance function that takes two time series as parameters. It should take
        the form distance_function(x: np.ndarray, y: np.ndarray).
    *args
        Additional positional arguments to pass to the distance function.
    **kwargs
        Additional keyword arguments to pass to the distance function.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        Distance matrix where [i, j] is the distance between the ith instance
        in x (x[i]) and the jth instance in y (y[j]).

    Example
    -------
    >>> from aeon.distances import euclidean_distance
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    >>> distance_from_single_to_multiple(x, y, euclidean_distance)
    """
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = distance_function(x[i], y[j], *args, **kwargs)
    return distances
