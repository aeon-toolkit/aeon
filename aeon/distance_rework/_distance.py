import numpy as np
from aeon.distance_rework import (
    euclidean_distance,
    squared_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    lcss_distance,
    msm_distance,
    erp_distance,
    edr_distance,
    twe_distance,
    euclidean_pairwise_distance,
    squared_pairwise_distance,
    dtw_pairwise_distance,
    ddtw_pairwise_distance,
    wdtw_pairwise_distance,
    wddtw_pairwise_distance,
    lcss_pairwise_distance,
    msm_pairwise_distance,
    erp_pairwise_distance,
    edr_pairwise_distance,
    twe_pairwise_distance,
    euclidean_from_single_to_multiple_distance,
    euclidean_from_multiple_to_multiple_distance,
    squared_from_single_to_multiple_distance,
    squared_from_multiple_to_multiple_distance,
    dtw_from_single_to_multiple_distance,
    dtw_from_multiple_to_multiple_distance,
    ddtw_from_single_to_multiple_distance,
    ddtw_from_multiple_to_multiple_distance,
    wdtw_from_single_to_multiple_distance,
    wdtw_from_multiple_to_multiple_distance,
    wddtw_from_single_to_multiple_distance,
    wddtw_from_multiple_to_multiple_distance,
    lcss_from_single_to_multiple_distance,
    lcss_from_multiple_to_multiple_distance,
    msm_from_single_to_multiple_distance,
    msm_from_multiple_to_multiple_distance,
    erp_from_single_to_multiple_distance,
    erp_from_multiple_to_multiple_distance,
    edr_from_single_to_multiple_distance,
    edr_from_multiple_to_multiple_distance,
    twe_from_single_to_multiple_distance,
    twe_from_multiple_to_multiple_distance,
    dtw_cost_matrix,
    ddtw_cost_matrix,
    wdtw_cost_matrix,
    wddtw_cost_matrix,
    lcss_cost_matrix,
    msm_cost_matrix,
    erp_cost_matrix,
    edr_cost_matrix,
    twe_cost_matrix,
)


def distance(x: np.ndarray, y: np.ndarray, metric: str, **kwargs):
    """Compute distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    metric : str
        Distance metric to use. Must be one of the following:
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
    float
        Distance between the two time series.

    Raises
    ------
    ValueError
        If the distance metric is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import distance
    >>> x = np.array([[1, 2, 3, 4, 5]])
    >>> y = np.array([[6, 7, 8, 9, 10]])
    >>> distance(x, y, "dtw", window = 0.2)
    125.0
    """
    if metric not in _distances:
        raise ValueError(f"Unknown distance metric: {metric}")
    return _distances[metric](x, y, **kwargs)


def pairwise_distance(X: np.ndarray, metric: str, **kwargs):
    """Compute pairwise distance between time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_dims, n_timepoints)
        Set of time series.
    metric : str
        Distance metric to use. Must be one of the following:
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
    >>> from aeon.distance_rework import pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> pairwise_distance(X, "dtw", window = 0.2)
    array([[  0.,  28., 109.],
           [ 28.,   0.,  27.],
           [109.,  27.,   0.]])
    """
    if metric not in _pairwise_distances:
        raise ValueError(f"Unknown distance metric: {metric}")
    return _pairwise_distances[metric](X, **kwargs)


def distance_from_single_to_multiple(
        x: np.ndarray, y: np.ndarray, metric: str, **kwargs
):
    """Compute the distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_dims, n_timepoints)
        A collection of time series instances.
    metric : str
        Distance metric to use. Must be one of the following:
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
    np.ndarray (n_instances)
        distance between the collection of instances in y and the time series x.

    Raises
    ------
    ValueError
        If the distance metric is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import distance_from_single_to_multiple
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> distance_from_single_to_multiple(x, y, "dtw", window=0.2)
    array([316., 532., 784.])
    """
    if metric not in _single_to_multiple_distances:
        raise ValueError(f"Unknown distance metric: {metric}")
    return _single_to_multiple_distances[metric](x, y, **kwargs)


def distance_from_multiple_to_multiple(
        x: np.ndarray, y: np.ndarray, metric: str, **kwargs
):
    """Compute the distance between two sets of time series.

    If x and y are the same then you should use pairwise_distance.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_dims, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_dims, n_timepoints)
        A collection of time series instances.
    metric : str
        Distance metric to use. Must be one of the following:
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
    >>> from aeon.distance_rework import distance_from_multiple_to_multiple
    >>> x = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_from_multiple_to_multiple_distance(x, y, window=0.2)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])
    """
    if metric not in _multiple_to_multiple_distances:
        raise ValueError(f"Unknown distance metric: {metric}")
    return _multiple_to_multiple_distances[metric](x, y, **kwargs)


def cost_matrix(x: np.ndarray, y: np.ndarray, metric: str, **kwargs):
    """Compute the cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    metric : str
        Distance metric to use. Must be one of the following:
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
    >>> from aeon.distance_rework import cost_matrix
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[11, 12, 13, 2]])
    >>> cost_matrix(x, y, "dtw", window=0.2)
    array([[100., 221.,  inf,  inf],
           [ inf, 200., 321.,  inf],
           [ inf,  inf, 300., 301.],
           [ inf,  inf,  inf, 316.]])
    """
    if metric not in _cost_matrices:
        raise ValueError(f"Unknown distance metric: {metric}")
    return _cost_matrices[metric](x, y, **kwargs)


_distances = {
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
    "twe": twe_distance
}

_pairwise_distances = {
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
    "twe": twe_pairwise_distance
}

_single_to_multiple_distances = {
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
    "twe": twe_from_single_to_multiple_distance
}

_multiple_to_multiple_distances = {
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
    "twe": twe_from_multiple_to_multiple_distance
}

_cost_matrices = {
    "dtw": dtw_cost_matrix,
    "ddtw": ddtw_cost_matrix,
    "wdtw": wdtw_cost_matrix,
    "wddtw": wddtw_cost_matrix,
    "lcss": lcss_cost_matrix,
    "msm": msm_cost_matrix,
    "erp": erp_cost_matrix,
    "edr": edr_cost_matrix,
    "twe": twe_cost_matrix
}
