# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any, Callable, List, Tuple, Union

import numpy as np

from aeon.distances._ddtw import (
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_pairwise_distance,
)
from aeon.distances._dtw import (
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_pairwise_distance,
)
from aeon.distances._edr import (
    edr_alignment_path,
    edr_cost_matrix,
    edr_distance,
    edr_pairwise_distance,
)
from aeon.distances._erp import (
    erp_alignment_path,
    erp_cost_matrix,
    erp_distance,
    erp_pairwise_distance,
)
from aeon.distances._euclidean import euclidean_distance, euclidean_pairwise_distance
from aeon.distances._lcss import (
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_distance,
    lcss_pairwise_distance,
)
from aeon.distances._msm import (
    msm_alignment_path,
    msm_cost_matrix,
    msm_distance,
    msm_pairwise_distance,
)
from aeon.distances._squared import squared_distance, squared_pairwise_distance
from aeon.distances._twe import (
    twe_alignment_path,
    twe_cost_matrix,
    twe_distance,
    twe_pairwise_distance,
)
from aeon.distances._utils import reshape_pairwise_to_multiple
from aeon.distances._wddtw import (
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_pairwise_distance,
)
from aeon.distances._wdtw import (
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_pairwise_distance,
)
from aeon.distances.mpdist import mpdist

DistanceFunction = Callable[[np.ndarray, np.ndarray, Any], float]
AlignmentPathFunction = Callable[
    [np.ndarray, np.ndarray, Any], Tuple[List[Tuple[int, int]], float]
]
CostMatrixFunction = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
PairwiseFunction = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]


def distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[str, DistanceFunction],
    **kwargs: Any,
) -> float:
    """Compute the distance between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp',
        'msm'
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a float.
    kwargs: Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Returns
    -------
    float
        Distance between the x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not a valid string or callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> distance(x, y, metric="dtw")
    768.0
    """
    if metric == "squared":
        return squared_distance(x, y)
    elif metric == "euclidean":
        return euclidean_distance(x, y)
    elif metric == "dtw":
        return dtw_distance(x, y, kwargs.get("window"))
    elif metric == "ddtw":
        return ddtw_distance(x, y, kwargs.get("window"))
    elif metric == "wdtw":
        return wdtw_distance(x, y, kwargs.get("window"), kwargs.get("g", 0.05))
    elif metric == "wddtw":
        return wddtw_distance(x, y, kwargs.get("window"), kwargs.get("g", 0.05))
    elif metric == "lcss":
        return lcss_distance(x, y, kwargs.get("window"), kwargs.get("epsilon", 1.0))
    elif metric == "erp":
        return erp_distance(
            x, y, kwargs.get("window"), kwargs.get("g", 0.0), kwargs.get("g_arr", None)
        )
    elif metric == "edr":
        return edr_distance(x, y, kwargs.get("window"), kwargs.get("epsilon"))
    elif metric == "twe":
        return twe_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
        )
    elif metric == "msm":
        return msm_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
        )
    elif metric == "mpdist":
        return mpdist(x, y, **kwargs)
    else:
        if isinstance(metric, Callable):
            return metric(x, y, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")


def pairwise_distance(
    x: np.ndarray,
    y: np.ndarray = None,
    metric: Union[str, DistanceFunction] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute the pairwise distance matrix between two time series.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,), default=None
        A collection of time series instances.
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp',
        'msm'
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a float.
    kwargs: Any
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.
        If metric is not a valid string or callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> pairwise_distance(X, metric='dtw')
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> pairwise_distance(X, y, metric='dtw')
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> pairwise_distance(X, y_univariate, metric='dtw')
    array([[300.],
           [147.],
           [ 48.]])
    """
    if metric == "squared":
        return squared_pairwise_distance(x, y)
    elif metric == "euclidean":
        return euclidean_pairwise_distance(x, y)
    elif metric == "dtw":
        return dtw_pairwise_distance(x, y, kwargs.get("window"))
    elif metric == "ddtw":
        return ddtw_pairwise_distance(x, y, kwargs.get("window"))
    elif metric == "wdtw":
        return wdtw_pairwise_distance(x, y, kwargs.get("window"), kwargs.get("g", 0.05))
    elif metric == "wddtw":
        return wddtw_pairwise_distance(
            x, y, kwargs.get("window"), kwargs.get("g", 0.05)
        )
    elif metric == "lcss":
        return lcss_pairwise_distance(
            x, y, kwargs.get("window"), kwargs.get("epsilon", 1.0)
        )
    elif metric == "erp":
        return erp_pairwise_distance(
            x, y, kwargs.get("window"), kwargs.get("g", 0.0), kwargs.get("g_arr", None)
        )
    elif metric == "edr":
        return edr_pairwise_distance(x, y, kwargs.get("window"), kwargs.get("epsilon"))
    elif metric == "twe":
        return twe_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
        )
    elif metric == "msm":
        return msm_pairwise_distance(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
        )
    elif metric == "mpdist":
        return _custom_func_pairwise(x, y, mpdist, **kwargs)
    else:
        if isinstance(metric, Callable):
            return _custom_func_pairwise(x, y, metric, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")


def _custom_func_pairwise(
    X: np.ndarray,
    y: np.ndarray = None,
    dist_func: DistanceFunction = None,
    **kwargs: Any,
) -> np.ndarray:
    if y is None:
        # To self
        if X.ndim == 3:
            return _custom_pairwise_distance(X, dist_func, **kwargs)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _custom_pairwise_distance(_X, dist_func, **kwargs)
        raise ValueError("x and y must be 2D or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _custom_from_multiple_to_multiple_distance(_x, _y, dist_func, **kwargs)


def _custom_pairwise_distance(
    X: np.ndarray, dist_func: DistanceFunction, **kwargs
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = dist_func(X[i], X[j], **kwargs)
            distances[j, i] = distances[i, j]

    return distances


def _custom_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, dist_func: DistanceFunction, **kwargs
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = dist_func(x[i], y[j], **kwargs)
    return distances


def alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    metric: str,
    **kwargs: Any,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the alignment path and distance between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    metric: str
        The distance metric to use. The value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp',
        'msm'
    kwargs: Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The dtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> alignment_path(x, y, metric='dtw')
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    if metric == "dtw":
        return dtw_alignment_path(x, y, kwargs.get("window"))
    elif metric == "ddtw":
        return ddtw_alignment_path(x, y, kwargs.get("window"))
    elif metric == "wdtw":
        return wdtw_alignment_path(x, y, kwargs.get("window"), kwargs.get("g", 0.05))
    elif metric == "wddtw":
        return wddtw_alignment_path(x, y, kwargs.get("window"), kwargs.get("g", 0.05))
    elif metric == "lcss":
        return lcss_alignment_path(
            x, y, kwargs.get("window"), kwargs.get("epsilon", 1.0)
        )
    elif metric == "erp":
        return erp_alignment_path(
            x, y, kwargs.get("window"), kwargs.get("g", 0.0), kwargs.get("g_arr", None)
        )
    elif metric == "edr":
        return edr_alignment_path(x, y, kwargs.get("window"), kwargs.get("epsilon"))
    elif metric == "twe":
        return twe_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
        )
    elif metric == "msm":
        return msm_alignment_path(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
        )
    else:
        raise ValueError("Metric must be one of the supported strings")


def cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    metric: str,
    **kwargs: Any,
) -> np.ndarray:
    """Compute the alignment path and distance between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    metric: str or Callable
        The distance metric to use. The value must be one of the following strings:
        'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp', 'msm'

    kwargs: Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.


    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> cost_matrix(x, y, metric="dtw")
    array([[  0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204., 285.],
           [  1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204.],
           [  5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140.],
           [ 14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91.],
           [ 30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.],
           [ 55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.],
           [ 91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.],
           [140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.],
           [204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.],
           [285., 204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.]])
    """
    if metric == "dtw":
        return dtw_cost_matrix(x, y, kwargs.get("window"))
    elif metric == "ddtw":
        return ddtw_cost_matrix(x, y, kwargs.get("window"))
    elif metric == "wdtw":
        return wdtw_cost_matrix(x, y, kwargs.get("window"), kwargs.get("g", 0.05))
    elif metric == "wddtw":
        return wddtw_cost_matrix(x, y, kwargs.get("window"), kwargs.get("g", 0.05))
    elif metric == "lcss":
        return lcss_cost_matrix(x, y, kwargs.get("window"), kwargs.get("epsilon", 1.0))
    elif metric == "erp":
        return erp_cost_matrix(
            x, y, kwargs.get("window"), kwargs.get("g", 0.0), kwargs.get("g_arr", None)
        )
    elif metric == "edr":
        return edr_cost_matrix(x, y, kwargs.get("window"), kwargs.get("epsilon"))
    elif metric == "twe":
        return twe_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("nu", 0.001),
            kwargs.get("lmbda", 1.0),
        )
    elif metric == "msm":
        return msm_cost_matrix(
            x,
            y,
            kwargs.get("window"),
            kwargs.get("independent", True),
            kwargs.get("c", 1.0),
        )
    else:
        raise ValueError("Metric must be one of the supported strings")


def get_distance_function(metric: Union[str, DistanceFunction]) -> DistanceFunction:
    """Get the distance function for a given metric string or callable.

    Parameters
    ----------
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp',
        'msm', 'mpdist'
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a float.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], float]
        The distance function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> from aeon.distances import get_distance_function
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dtw_dist_func = get_distance_function("dtw")
    >>> dtw_dist_func(x, y, window=0.2)
    874.0
    """
    return _resolve_key_from_distance(metric, "distance")


def get_pairwise_distance_function(
    metric: Union[str, PairwiseFunction]
) -> PairwiseFunction:
    """Get the pairwise distance function for a given metric string or callable.

    Parameters
    ----------
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp',
        'msm'
        If a callable is given, the value must be a function that accepts two
        numpy arrays and **kwargs returns a np.ndarray that is the pairwise distance
        between each time series.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
        The pairwise distance function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.

    Examples
    --------
    >>> from aeon.distances import get_pairwise_distance_function
    >>> import numpy as np
    >>> x = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_pairwise_dist_func = get_pairwise_distance_function("dtw")
    >>> dtw_pairwise_dist_func(x, y, window=0.2)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])
    """
    return _resolve_key_from_distance(metric, "pairwise_distance")


def get_alignment_path_function(metric: str) -> AlignmentPathFunction:
    """Get the alignment path function for a given metric string or callable.

    Parameters
    ----------
    metric: str or Callable
        The distance metric to use. The value must be one of the following strings:
        'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp', 'msm'

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], Tuple[List[Tuple[int, int]], float]]
        The alignment path function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.
        If the metric doesn't have an alignment path function.

    Examples
    --------
    >>> from aeon.distances import get_alignment_path_function
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 4, 5]])
    >>> y = np.array([[11, 12, 13, 14, 15]])
    >>> dtw_alignment_path_func = get_alignment_path_function("dtw")
    >>> dtw_alignment_path_func(x, y, window=0.2)
    ([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], 500.0)
    """
    return _resolve_key_from_distance(metric, "alignment_path")


def get_cost_matrix_function(metric: str) -> CostMatrixFunction:
    """Get the cost matrix function for a given metric string or callable.

    Parameters
    ----------
    metric: str or Callable
        The distance metric to use. The value must be one of the following strings:
        'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp', 'msm'
        two time series.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
        The cost matrix function for the given metric.

    Raises
    ------
    ValueError
        If metric is not one of the supported strings or a callable.
        If the metric doesn't have a cost matrix function.

    Examples
    --------
    >>> from aeon.distances import get_cost_matrix_function
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 4, 5]])
    >>> y = np.array([[11, 12, 13, 14, 15]])
    >>> dtw_cost_matrix_func = get_cost_matrix_function("dtw")
    >>> dtw_cost_matrix_func(x, y, window=0.2)
    array([[100., 221.,  inf,  inf,  inf],
           [181., 200., 321.,  inf,  inf],
           [ inf, 262., 300., 421.,  inf],
           [ inf,  inf, 343., 400., 521.],
           [ inf,  inf,  inf, 424., 500.]])
    """
    return _resolve_key_from_distance(metric, "cost_matrix")


def _resolve_key_from_distance(metric: str, key: str) -> Any:
    if metric == "mpdist":
        return mpdist
    dist = DISTANCES_DICT.get(metric)
    if dist is None:
        raise ValueError(f"Unknown metric {metric}")
    dist_callable = dist.get(key)
    if dist_callable is None:
        raise ValueError(f"Metric {metric} does not have a {key} function")
    return dist_callable


DISTANCES = [
    {
        "name": "euclidean",
        "distance": euclidean_distance,
        "pairwise_distance": euclidean_pairwise_distance,
    },
    {
        "name": "squared",
        "distance": squared_distance,
        "pairwise_distance": squared_pairwise_distance,
    },
    {
        "name": "dtw",
        "distance": dtw_distance,
        "pairwise_distance": dtw_pairwise_distance,
        "cost_matrix": dtw_cost_matrix,
        "alignment_path": dtw_alignment_path,
    },
    {
        "name": "ddtw",
        "distance": ddtw_distance,
        "pairwise_distance": ddtw_pairwise_distance,
        "cost_matrix": ddtw_cost_matrix,
        "alignment_path": ddtw_alignment_path,
    },
    {
        "name": "wdtw",
        "distance": wdtw_distance,
        "pairwise_distance": wdtw_pairwise_distance,
        "cost_matrix": wdtw_cost_matrix,
        "alignment_path": wdtw_alignment_path,
    },
    {
        "name": "wddtw",
        "distance": wddtw_distance,
        "pairwise_distance": wddtw_pairwise_distance,
        "cost_matrix": wddtw_cost_matrix,
        "alignment_path": wddtw_alignment_path,
    },
    {
        "name": "lcss",
        "distance": lcss_distance,
        "pairwise_distance": lcss_pairwise_distance,
        "cost_matrix": lcss_cost_matrix,
        "alignment_path": lcss_alignment_path,
    },
    {
        "name": "erp",
        "distance": erp_distance,
        "pairwise_distance": erp_pairwise_distance,
        "cost_matrix": erp_cost_matrix,
        "alignment_path": erp_alignment_path,
    },
    {
        "name": "edr",
        "distance": edr_distance,
        "pairwise_distance": edr_pairwise_distance,
        "cost_matrix": edr_cost_matrix,
        "alignment_path": edr_alignment_path,
    },
    {
        "name": "twe",
        "distance": twe_distance,
        "pairwise_distance": twe_pairwise_distance,
        "cost_matrix": twe_cost_matrix,
        "alignment_path": twe_alignment_path,
    },
    {
        "name": "msm",
        "distance": msm_distance,
        "pairwise_distance": msm_pairwise_distance,
        "cost_matrix": msm_cost_matrix,
        "alignment_path": msm_alignment_path,
    },
]

DISTANCES_DICT = {d["name"]: d for d in DISTANCES}
