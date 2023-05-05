# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any, Callable, Union

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
from aeon.distances._msm import _MsmDistance
from aeon.distances._numba_utils import (
    _compute_pairwise_distance,
    _make_3d_series,
    _numba_to_timeseries,
    to_numba_timeseries,
)
from aeon.distances._resolve_metric import (
    _resolve_dist_instance,
    _resolve_metric_to_factory,
)
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
from aeon.distances.base import (
    AlignmentPathReturn,
    DistanceAlignmentPathCallable,
    DistanceCallable,
    MetricInfo,
    NumbaDistance,
)


def msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    window: float = None,
    **kwargs: dict,
) -> float:
    """Compute the move-split-merge distance.

    This metric uses as building blocks three fundamental operations: Move, Split,
    and Merge. A Move operation changes the value of a single element, a Split
    operation converts a single element into two consecutive elements, and a Merge
    operation merges two consecutive elements into one. Each operation has an
    associated cost, and the MSM distance between two time series is defined to be
    the cost of the cheapest sequence of operations that transforms the first time
    series into the second one.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First time series.
    y: np.ndarray (1d or 2d array)
        Second time series.
    c: float, default = 1.0
        Cost for split or merge operation.
    window: Float, defaults = None
        Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding). Must be between 0 and 1.
    kwargs: any
        extra kwargs.

    Returns
    -------
    float
        Msm distance between x and y.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined
    References
    ----------
    .. [1]A.  Stefan,  V.  Athitsos,  and  G.  Das.   The  Move-Split-Merge  metric
    for time  series. IEEE  Transactions  on  Knowledge  and  Data  Engineering,
    25(6):1425–1438, 2013.
    """
    format_kwargs = {
        "c": c,
        "window": window,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance(x, y, metric="msm", **format_kwargs)


def msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    return_cost_matrix: bool = False,
    c: float = 1.0,
    window: float = None,
    **kwargs: dict,
) -> AlignmentPathReturn:
    """Compute the move-split-merge alignment path.

    This metric uses as building blocks three fundamental operations: Move, Split,
    and Merge. A Move operation changes the value of a single element, a Split
    operation converts a single element into two consecutive elements, and a Merge
    operation merges two consecutive elements into one. Each operation has an
    associated cost, and the MSM distance between two time series is defined to be
    the cost of the cheapest sequence of operations that transforms the first time
    series into the second one.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        First time series.
    y: np.ndarray (1d or 2d array)
        Second time series.
    return_cost_matrix: bool, defaults = False
        Boolean that when true will also return the cost matrix.
    c: float, default = 1.0
        Cost for split or merge operation.
    window: float, defaults = None
        Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding). Must be between 0 and 1.
    kwargs: any
        extra kwargs.

    Returns
    -------
    list[tuple]
        List of tuples containing the msm alignment path.
    float
        Msm distance between x and y.
    np.ndarray (of shape (len(x), len(y)).
        Optional return only given if return_cost_matrix = True.
        Cost matrix used to compute the distance.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 2 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined
    References
    ----------
    .. [1]A.  Stefan,  V.  Athitsos,  and  G.  Das.   The  Move-Split-Merge  metric
    for time  series. IEEE  Transactions  on  Knowledge  and  Data  Engineering,
    25(6):1425–1438, 2013.
    """
    format_kwargs = {
        "c": c,
        "window": window,
    }
    format_kwargs = {**format_kwargs, **kwargs}

    return distance_alignment_path(
        x, y, metric="msm", return_cost_matrix=return_cost_matrix, **format_kwargs
    )


NEW_DISTANCES = [
    "squared",
    "euclidean",
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "lcss",
    "erp",
    "edr",
    "twe",
]


def distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
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
    if metric in NEW_DISTANCES:
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
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)

    _metric_callable = _resolve_metric_to_factory(
        metric, _x, _y, _METRIC_INFOS, **kwargs
    )

    return _metric_callable(_x, _y)


def distance_factory(
    x: np.ndarray = None,
    y: np.ndarray = None,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ] = "euclidean",
    **kwargs: Any,
) -> DistanceCallable:
    """Create a no_python distance callable.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array), defaults = None
        First time series.
    y: np.ndarray (1d or 2d array), defaults = None
        Second time series.
    metric: str or Callable, defaults  = 'euclidean'
        The distance metric to use.
        If a string is given, the value must be one of the following strings:
        'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp'

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

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]]
        No_python compiled distance callable.

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
    """
    if metric in NEW_DISTANCES:
        if metric == "squared":
            return squared_distance
        elif metric == "euclidean":
            return euclidean_distance
        elif metric == "dtw":
            return dtw_distance
        elif metric == "ddtw":
            return ddtw_distance
        elif metric == "wdtw":
            return wdtw_distance
        elif metric == "wddtw":
            return wddtw_distance
        elif metric == "lcss":
            return lcss_distance
        elif metric == "erp":
            return erp_distance
        elif metric == "edr":
            return edr_distance
        elif metric == "twe":
            return twe_distance
    global dist_callable

    if x is None:
        x = np.zeros((1, 10))
    if y is None:
        y = np.zeros((1, 10))
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)

    callable = _resolve_metric_to_factory(metric, _x, _y, _METRIC_INFOS, **kwargs)

    # TODO Not sure why, but removing this @njit, avoids recompiling the closures
    # @njit(cache=True)
    def dist_callable(x: np.ndarray, y: np.ndarray):
        _x = _numba_to_timeseries(x)
        _y = _numba_to_timeseries(y)
        return callable(_x, _y)

    return dist_callable


def pairwise_distance(
    x: np.ndarray,
    y: np.ndarray = None,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
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
    _x = _make_3d_series(x)
    if y is None:
        y = x
    _y = _make_3d_series(y)
    if metric in NEW_DISTANCES:
        if metric == "euclidean":
            return euclidean_pairwise_distance(_x, _y)
        elif metric == "squared":
            return squared_pairwise_distance(_x, _y)
        elif metric == "dtw":
            return dtw_pairwise_distance(_x, _y, **kwargs)
        elif metric == "ddtw":
            return ddtw_pairwise_distance(_x, _y, **kwargs)
        elif metric == "wdtw":
            return wdtw_pairwise_distance(_x, _y, **kwargs)
        elif metric == "wddtw":
            return wddtw_pairwise_distance(_x, _y, **kwargs)
        elif metric == "lcss":
            return lcss_pairwise_distance(_x, _y, **kwargs)
        elif metric == "erp":
            return erp_pairwise_distance(_x, _y, **kwargs)
        elif metric == "edr":
            return edr_pairwise_distance(_x, _y, **kwargs)
        elif metric == "twe":
            return twe_pairwise_distance(_x, _y, **kwargs)

    symmetric = np.array_equal(_x, _y)
    _metric_callable = _resolve_metric_to_factory(
        metric, _x[0], _y[0], _METRIC_INFOS, **kwargs
    )
    return _compute_pairwise_distance(_x, _y, symmetric, _metric_callable)


def distance_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ],
    return_cost_matrix: bool = False,
    **kwargs: Any,
) -> AlignmentPathReturn:
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

    Returns
    -------
    list[tuple]
        List of tuples containing the alginment path for the distance.
    float
        Distance between the x and y.
    np.ndarray (of shape (len(x), len(y)).
        Optional return only given if return_cost_matrix = True.
        Cost matrix used to compute the distance.
    """
    if metric in NEW_DISTANCES:
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
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)

    _dist_instance = _resolve_dist_instance(metric, _x, _y, _METRIC_INFOS, **kwargs)

    return _dist_instance.distance_alignment_path(
        _x, _y, return_cost_matrix=return_cost_matrix, **kwargs
    )


def distance_alignment_path_factory(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ],
    return_cost_matrix: bool = False,
    **kwargs: Any,
) -> DistanceAlignmentPathCallable:
    """Produce a distance alignment path factory numba callable.

    First the distance metric is 'resolved'. This means the metric that is passed
    is resolved to callable. The callable is then called with x and y and the
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

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], Union[np.ndarray, np.ndarray]]
        Callable for the distance path.
    """
    if metric in NEW_DISTANCES:
        if metric == "dtw":
            return dtw_alignment_path
        elif metric == "ddtw":
            return ddtw_alignment_path
        elif metric == "wdtw":
            return wdtw_alignment_path
        elif metric == "wddtw":
            return wddtw_alignment_path
        elif metric == "lcss":
            return lcss_alignment_path
        elif metric == "erp":
            return erp_alignment_path
        elif metric == "edr":
            return edr_alignment_path
        elif metric == "twe":
            return twe_alignment_path
    if x is None:
        x = np.zeros((1, 10))
    if y is None:
        y = np.zeros((1, 10))
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)

    dist_instance = _resolve_dist_instance(metric, _x, _y, _METRIC_INFOS, **kwargs)
    callable = dist_instance.distance_alignment_path_factory(
        _x, _y, return_cost_matrix, **kwargs
    )

    @njit(cache=True)
    def dist_callable(x: np.ndarray, y: np.ndarray):
        _x = _numba_to_timeseries(x)
        _y = _numba_to_timeseries(y)
        return callable(_x, _y)

    return dist_callable


_METRIC_INFOS = [
    MetricInfo(
        canonical_name="msm",
        aka={"msm", "move-split-merge"},
        dist_func=msm_distance,
        dist_instance=_MsmDistance(),
        dist_alignment_path_func=msm_alignment_path,
    ),
]

_METRICS = {info.canonical_name: info for info in _METRIC_INFOS}
_METRIC_ALIAS = dict((alias, info) for info in _METRIC_INFOS for alias in info.aka)
_METRIC_CALLABLES = dict(
    (info.canonical_name, info.dist_func) for info in _METRIC_INFOS
)
_METRICS_NAMES = list(_METRICS.keys())

ALL_DISTANCES = (msm_distance,)
