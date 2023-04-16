# -*- coding: utf-8 -*-
from typing import Any, Callable, List, Tuple, Union

import numpy as np

from aeon.distances._dtw import (
    dtw_distance,
    dtw_cost_matrix,
    dtw_from_multiple_to_multiple_distance,
    dtw_from_single_to_multiple_distance,
    dtw_pairwise_distance,
    dtw_alignment_path
)
from aeon.distances._ddtw import (
    ddtw_distance,
    ddtw_cost_matrix,
    ddtw_from_multiple_to_multiple_distance,
    ddtw_from_single_to_multiple_distance,
    ddtw_pairwise_distance,
    ddtw_alignment_path
)
from aeon.distances._wdtw import (
    wdtw_distance,
    wdtw_cost_matrix,
    wdtw_from_multiple_to_multiple_distance,
    wdtw_from_single_to_multiple_distance,
    wdtw_pairwise_distance,
    wdtw_alignment_path
)
from aeon.distances._wddtw import (
    wddtw_distance,
    wddtw_cost_matrix,
    wddtw_from_multiple_to_multiple_distance,
    wddtw_from_single_to_multiple_distance,
    wddtw_pairwise_distance,
    wddtw_alignment_path
)
from aeon.distances._lcss import (
    lcss_distance,
    lcss_cost_matrix,
    lcss_from_multiple_to_multiple_distance,
    lcss_from_single_to_multiple_distance,
    lcss_pairwise_distance,
    lcss_alignment_path
)
from aeon.distances._msm import (
    msm_distance,
    msm_cost_matrix,
    msm_from_multiple_to_multiple_distance,
    msm_from_single_to_multiple_distance,
    msm_pairwise_distance,
    msm_alignment_path
)
from aeon.distances._erp import (
    erp_distance,
    erp_cost_matrix,
    erp_from_multiple_to_multiple_distance,
    erp_from_single_to_multiple_distance,
    erp_pairwise_distance,
    erp_alignment_path
)
from aeon.distances._edr import (
    edr_distance,
    edr_cost_matrix,
    edr_from_multiple_to_multiple_distance,
    edr_from_single_to_multiple_distance,
    edr_pairwise_distance,
    edr_alignment_path
)
from aeon.distances._twe import (
    twe_distance,
    twe_cost_matrix,
    twe_from_multiple_to_multiple_distance,
    twe_from_single_to_multiple_distance,
    twe_pairwise_distance,
    twe_alignment_path
)
from aeon.distances._euclidean import (
    euclidean_distance,
    euclidean_from_multiple_to_multiple_distance,
    euclidean_from_single_to_multiple_distance,
    euclidean_pairwise_distance
)
from aeon.distances._squared import (
    squared_distance,
    squared_from_multiple_to_multiple_distance,
    squared_from_single_to_multiple_distance,
    squared_pairwise_distance
)


MetricCallableType = Union[Callable[[np.ndarray, np.ndarray, Any], float], str]


def distance(x: np.ndarray, y: np.ndarray, metric: MetricCallableType, **kwargs):
    """Compute distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    metric : str or Callable[[np.ndarray, np.ndarray, **kwargs], float]
        Distance metric to use. If a string Must be one of the following:
        - "euclidean"
        - "squared"
        - "dtw"
        - "ddtw"
        - "wdtw"
        - "wddtw"
        - "lcss"
        - "msm"
        - "erp"
        - "edr"
        - "twe"
        If a callable, it must take in two time series and return a float.
    **kwargs
        Additional keyword arguments to pass to the distance function. See the distance
        function documentation for more details on what to pass.

    Returns
    -------
    float
        Distance between the two time series.

    Raises
    ------
    ValueError
        If the distance metric is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import distance
    >>> x = np.array([[1, 2, 3, 4, 5]])
    >>> y = np.array([[6, 7, 8, 9, 10]])
    >>> distance(x, y, "dtw", window = 0.2)
    125.0
    """
    if isinstance(metric, Callable):
        return metric(x, y, **kwargs)
    if metric not in distance_function_dict:
        raise ValueError(f"Unknown distance metric: {metric}")
    return distance_function_dict[metric](x, y, **kwargs)


def pairwise_distance(X: np.ndarray, metric: str, **kwargs):
    """Compute pairwise distance between time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_channels, n_timepoints)
        Set of time series.
    metric : str
        Distance metric to use. If a string Must be one of the following:
        - "euclidean"
        - "squared"
        - "dtw"
        - "ddtw"
        - "wdtw"
        - "wddtw"
        - "lcss"
        - "msm"
        - "erp"
        - "edr"
        - "twe"
    **kwargs
        Additional keyword arguments to pass to the distance function. See the distance
        function documentation for more details on what to pass.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        dtw pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If the distance metric is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> pairwise_distance(X, "dtw", window = 0.2)
    array([[  0.,  28., 109.],
           [ 28.,   0.,  27.],
           [109.,  27.,   0.]])
    """
    if metric not in pairwise_distance_function_dict:
        raise ValueError(f"Unknown distance metric: {metric}")
    return pairwise_distance_function_dict[metric](X, **kwargs)


def distance_from_single_to_multiple(
        x: np.ndarray, y: np.ndarray, metric: str, **kwargs
):
    """Compute the distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    metric : str
        Distance metric to use. If a string Must be one of the following:
        - "euclidean"
        - "squared"
        - "dtw"
        - "ddtw"
        - "wdtw"
        - "wddtw"
        - "lcss"
        - "msm"
        - "erp"
        - "edr"
        If a callable, it must take in two time series and return a float.
    **kwargs
        Additional keyword arguments to pass to the distance function. See the distance
        function documentation for more details on what to pass.

    Returns
    -------
    np.ndarray (n_instances)
        distance between the collection of instances in y and the time series x.

    Raises
    ------
    ValueError
        If the distance metric is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import distance_from_single_to_multiple
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> distance_from_single_to_multiple(x, y, "dtw", window=0.2)
    array([316., 532., 784.])
    """
    if metric not in single_to_multiple_distance_function_dict:
        raise ValueError(f"Unknown distance metric: {metric}")
    return single_to_multiple_distance_function_dict[metric](x, y, **kwargs)


def distance_from_multiple_to_multiple(
        x: np.ndarray, y: np.ndarray, metric: str, **kwargs
):
    """Compute the distance between two sets of time series.

    If x and y are the same then you should use pairwise_distance.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_channels, n_timepoints)
        A collection of time series instances.
    metric : str
        Distance metric to use. If a string Must be one of the following:
        - "euclidean"
        - "squared"
        - "dtw"
        - "ddtw"
        - "wdtw"
        - "wddtw"
        - "lcss"
        - "msm"
        - "erp"
        - "edr"
        - "twe"
    **kwargs
        Additional keyword arguments to pass to the distance function. See the distance
        function documentation for more details on what to pass.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        distance between two collections of time series, x and y.

    Raises
    ------
    ValueError
        If the distance metric is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import distance_from_multiple_to_multiple
    >>> x = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_from_multiple_to_multiple_distance(x, y, window=0.2)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])
    """
    if metric not in multiple_to_multiple_distance_function_dict:
        raise ValueError(f"Unknown distance metric: {metric}")

    return multiple_to_multiple_distance_function_dict[metric](x, y, **kwargs)


def cost_matrix(
        x: np.ndarray, y: np.ndarray, metric: MetricCallableType, **kwargs
) -> np.ndarray:
    """Compute the cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    metric : str
        Distance metric to use. If a string Must be one of the following:
        - "dtw"
        - "ddtw"
        - "wdtw"
        - "wddtw"
        - "lcss"
        - "msm"
        - "erp"
        - "edr"
        - "twe"
    **kwargs
        Additional keyword arguments to pass to the distance function. See the distance
        function documentation for more details on what to pass.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        Cost matrix between the two time series.

    Raises
    ------
    ValueError
        If the distance metric is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import cost_matrix
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[11, 12, 13, 2]])
    >>> cost_matrix(x, y, "dtw", window=0.2)
    array([[100., 221.,  inf,  inf],
           [ inf, 200., 321.,  inf],
           [ inf,  inf, 300., 301.],
           [ inf,  inf,  inf, 316.]])
    """
    if metric not in cost_matrix_function_dict:
        raise ValueError(f"Unknown distance metric: {metric}")
    return cost_matrix_function_dict[metric](x, y, **kwargs)


def alignment_path(
        x: np.ndarray, y: np.ndarray, metric: str, **kwargs
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    metric : str
        Distance metric to use. If a string Must be one of the following:
        - "dtw"
        - "ddtw"
        - "wdtw"
        - "wddtw"
        - "lcss"
        - "msm"
        - "erp"
        - "edr"
        - "twe"
    **kwargs
        Additional keyword arguments to pass to the distance function. See the distance
        function documentation for more details on what to pass.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The wdtw distance betweeen the two time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[11, 12, 13, 2]])
    >>> alignment_path(x, y, "dtw", window=0.2)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 316.0)
    """
    if metric not in alignment_path_function_dict:
        raise ValueError(f"Unknown distance metric: {metric}")
    return alignment_path_function_dict[metric](x, y, **kwargs)


distance_function_dict = {
    "euclidean": euclidean_distance,
    "squared": squared_distance,
    "dtw": dtw_distance,
    "ddtw": ddtw_distance,
    "wdtw": wdtw_distance,
    "wddtw": wddtw_distance,
    "lcss": lcss_distance,
    "msm": msm_distance,
    "erp": erp_distance,
    "edr": edr_distance,
    "twe": twe_distance,
}

pairwise_distance_function_dict = {
    "euclidean": euclidean_pairwise_distance,
    "squared": squared_pairwise_distance,
    "dtw": dtw_pairwise_distance,
    "ddtw": ddtw_pairwise_distance,
    "wdtw": wdtw_pairwise_distance,
    "wddtw": wddtw_pairwise_distance,
    "lcss": lcss_pairwise_distance,
    "msm": msm_pairwise_distance,
    "erp": erp_pairwise_distance,
    "edr": edr_pairwise_distance,
    "twe": twe_pairwise_distance,
}

single_to_multiple_distance_function_dict = {
    "euclidean": euclidean_from_single_to_multiple_distance,
    "squared": squared_from_single_to_multiple_distance,
    "dtw": dtw_from_single_to_multiple_distance,
    "ddtw": ddtw_from_single_to_multiple_distance,
    "wdtw": wdtw_from_single_to_multiple_distance,
    "wddtw": wddtw_from_single_to_multiple_distance,
    "lcss": lcss_from_single_to_multiple_distance,
    "msm": msm_from_single_to_multiple_distance,
    "erp": erp_from_single_to_multiple_distance,
    "edr": edr_from_single_to_multiple_distance,
    "twe": twe_from_single_to_multiple_distance,
}

multiple_to_multiple_distance_function_dict = {
    "euclidean": euclidean_from_multiple_to_multiple_distance,
    "squared": squared_from_multiple_to_multiple_distance,
    "dtw": dtw_from_multiple_to_multiple_distance,
    "ddtw": ddtw_from_multiple_to_multiple_distance,
    "wdtw": wdtw_from_multiple_to_multiple_distance,
    "wddtw": wddtw_from_multiple_to_multiple_distance,
    "lcss": lcss_from_multiple_to_multiple_distance,
    "msm": msm_from_multiple_to_multiple_distance,
    "erp": erp_from_multiple_to_multiple_distance,
    "edr": edr_from_multiple_to_multiple_distance,
    "twe": twe_from_multiple_to_multiple_distance,
}

cost_matrix_function_dict = {
    "dtw": dtw_cost_matrix,
    "ddtw": ddtw_cost_matrix,
    "wdtw": wdtw_cost_matrix,
    "wddtw": wddtw_cost_matrix,
    "lcss": lcss_cost_matrix,
    "msm": msm_cost_matrix,
    "erp": erp_cost_matrix,
    "edr": edr_cost_matrix,
    "twe": twe_cost_matrix,
}

alignment_path_function_dict = {
    "dtw": dtw_alignment_path,
    "ddtw": ddtw_alignment_path,
    "wdtw": wdtw_alignment_path,
    "wddtw": wddtw_alignment_path,
    "lcss": lcss_alignment_path,
    "msm": msm_alignment_path,
    "erp": erp_alignment_path,
    "edr": edr_alignment_path,
    "twe": twe_alignment_path,
}
