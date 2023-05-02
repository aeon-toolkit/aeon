# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any, Callable, Union, List, Tuple

import numpy as np
from numba import njit

from aeon.distances._ddtw import (
    ddtw_alignment_path,
    ddtw_distance,
    ddtw_pairwise_distance,
)
from aeon.distances._dtw import dtw_alignment_path, dtw_distance, dtw_pairwise_distance
from aeon.distances._edr import edr_alignment_path, edr_distance, edr_pairwise_distance
from aeon.distances._erp import erp_alignment_path, erp_distance, erp_pairwise_distance
from aeon.distances._euclidean import euclidean_distance, euclidean_pairwise_distance
from aeon.distances._lcss import (
    lcss_alignment_path,
    lcss_distance,
    lcss_pairwise_distance,
)
from aeon.distances._msm import msm_alignment_path, msm_distance, msm_pairwise_distance
from aeon.distances._squared import squared_distance, squared_pairwise_distance
from aeon.distances._twe import twe_alignment_path, twe_distance, twe_pairwise_distance
from aeon.distances._wddtw import (
    wddtw_alignment_path,
    wddtw_distance,
    wddtw_pairwise_distance,
)
from aeon.distances._wdtw import (
    wdtw_alignment_path,
    wdtw_distance,
    wdtw_pairwise_distance,
)


def distance(
        x: np.ndarray,
        y: np.ndarray,
        metric: Union[
            str,
            Callable[[np.ndarray, np.ndarray, Any], float],
        ],
        **kwargs: Any,
) -> float:
    """Compute the distance between two time series.

    First the distance metric is 'resolved'. This means the metric that is passed
    is resolved to a callable. The callable is then called with x and y and the
    value is then returned.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First time series.
    y: np.ndarray (1d or 2d array)
        Second time series.
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp',
        'msm'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    kwargs: Any
        Arguments for metric. Refer to each metrics documentation for a list of
        possible arguments.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.

    Examples
    --------
    >>> import numpy as np
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> distance(x_1d, y_1d, metric='dtw')
    58.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> distance(x_2d, y_2d, metric='dtw')
    512.0

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> distance(x_2d, y_2d, metric='dtw', window=0.5)
    512.0

    Returns
    -------
    float
        Distance between the x and y.
    """
    if metric == "squared":
        return squared_distance(x, y)
    elif metric == "euclidean":
        return euclidean_distance(x, y)
    elif metric == "dtw":
        return dtw_distance(x, y, **kwargs)
    elif metric == "ddtw":
        return ddtw_distance(x, y, **kwargs)
    elif metric == "wdtw":
        return wdtw_distance(x, y, **kwargs)
    elif metric == "wddtw":
        return wddtw_distance(x, y, **kwargs)
    elif metric == "lcss":
        return lcss_distance(x, y, **kwargs)
    elif metric == "erp":
        return erp_distance(x, y, **kwargs)
    elif metric == "edr":
        return edr_distance(x, y, **kwargs)
    elif metric == "twe":
        return twe_distance(x, y, **kwargs)
    elif metric == "msm":
        return msm_distance(x, y, **kwargs)
    else:
        if isinstance(metric, Callable):
            return metric(x, y, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")


def pairwise_distance(
        x: np.ndarray,
        y: np.ndarray = None,
        metric: Union[
            str,
            Callable[[np.ndarray, np.ndarray, Any], float],
        ] = "euclidean",
        **kwargs: Any,
) -> np.ndarray:
    """Compute the pairwise distance matrix between two time series.

    First the distance metric is 'resolved'. This means the metric that is passed
    is resolved to a callable. The callable is then called with x and y and the
    value is then returned. Then for each combination of x and y, the distance between
    the values are computed resulting in a 2d pairwise matrix.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First time series.
    y: np.ndarray (1d, 2d or 3d array), defaults = None
        Second time series. If not specified then y is set to the value of x.
    metric: str or Callable, defaults = 'euclidean'
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
            'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw',
            'lcss', 'edr', 'erp', 'msm'
        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    kwargs: Any
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
        Pairwise distance matrix between the two time series.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.

    Examples
    --------
    >>> import numpy as np
    >>> x_1d = np.array([1, 2, 3, 4])  # 1d array
    >>> y_1d = np.array([5, 6, 7, 8])  # 1d array
    >>> pairwise_distance(x_1d, y_1d, metric='dtw')
    array([[16., 25., 36., 49.],
           [ 9., 16., 25., 36.],
           [ 4.,  9., 16., 25.],
           [ 1.,  4.,  9., 16.]])

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> pairwise_distance(x_2d, y_2d, metric='dtw')
    array([[256., 576.],
           [ 58., 256.]])

    >>> x_3d = np.array([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])  # 3d array
    >>> y_3d = np.array([[[9], [10], [11], [12]], [[13], [14], [15], [16]]])  # 3d array
    >>> pairwise_distance(x_3d, y_3d, metric='dtw')
    array([[256., 576.],
           [ 64., 256.]])

    >>> x_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2d array
    >>> y_2d = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2d array
    >>> pairwise_distance(x_2d, y_2d, metric='dtw', window=0.5)
    array([[256., 576.],
           [ 58., 256.]])
    """
    if metric == "euclidean":
        return euclidean_pairwise_distance(x, y)
    elif metric == "squared":
        return squared_pairwise_distance(x, y)
    elif metric == "dtw":
        return dtw_pairwise_distance(x, y, **kwargs)
    elif metric == "ddtw":
        return ddtw_pairwise_distance(x, y, **kwargs)
    elif metric == "wdtw":
        return wdtw_pairwise_distance(x, y, **kwargs)
    elif metric == "wddtw":
        return wddtw_pairwise_distance(x, y, **kwargs)
    elif metric == "lcss":
        return lcss_pairwise_distance(x, y, **kwargs)
    elif metric == "erp":
        return erp_pairwise_distance(x, y, **kwargs)
    elif metric == "edr":
        return edr_pairwise_distance(x, y, **kwargs)
    elif metric == "twe":
        return twe_pairwise_distance(x, y, **kwargs)
    elif metric == "msm":
        return msm_pairwise_distance(x, y, **kwargs)
    else:
        if isinstance(metric, Callable):
            return metric(x, y, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")


def distance_alignment_path(
        x: np.ndarray,
        y: np.ndarray,
        metric: Union[
            str,
            Callable[[np.ndarray, np.ndarray, Any], float],
        ],
        return_cost_matrix: bool = False,
        **kwargs: Any,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the alignment path and distance between two time series.

    First the distance metric is 'resolved'. This means the metric that is passed
    is resolved to a callable. The callable is then called with x and y and the
    value is then returned.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First time series.
    y: np.ndarray (1d or 2d array)
        Second time series.
    metric: str or Callable
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp',
        'msm'

        If callable then it has to be a distance factory or numba distance callable.
        If you want to pass custom kwargs to the distance at runtime, use a distance
        factory as it constructs the distance using the kwargs before distance
        computation.
        A distance callable takes the form (must be no_python compiled):
        Callable[[np.ndarray, np.ndarray], float]

        A distance factory takes the form (must return a no_python callable):
        Callable[[np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray,
        np.ndarray], float]].
    return_cost_matrix: bool, defaults = False
        Boolean that when true will also return the cost matrix.
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
        The distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If metric is not a valid string or callable.

    """
    if metric == "dtw":
        return dtw_alignment_path(x, y, **kwargs)
    elif metric == "ddtw":
        return ddtw_alignment_path(x, y, **kwargs)
    elif metric == "wdtw":
        return wdtw_alignment_path(x, y, **kwargs)
    elif metric == "wddtw":
        return wddtw_alignment_path(x, y, **kwargs)
    elif metric == "lcss":
        return lcss_alignment_path(x, y, **kwargs)
    elif metric == "erp":
        return erp_alignment_path(x, y, **kwargs)
    elif metric == "edr":
        return edr_alignment_path(x, y, **kwargs)
    elif metric == "twe":
        return twe_alignment_path(x, y, **kwargs)
    elif metric == "msm":
        return msm_alignment_path(x, y, **kwargs)
    else:
        if isinstance(metric, Callable):
            return metric(x, y, **kwargs)
        raise ValueError("Metric must be one of the supported strings or a callable")
