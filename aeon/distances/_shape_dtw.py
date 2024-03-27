r"""Shape Dynamic time warping (ShapeDTW) between two time series."""

__maintainer__ = []

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._dtw import _dtw_cost_matrix
from aeon.distances._squared import _univariate_squared_distance
from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def _pad_ts_edges(x: np.ndarray, reach: int) -> np.ndarray:
    if x.ndim == 2:
        n_channels = x.shape[0]
        n_timepoints = x.shape[1]

        n_channels = int(n_channels)
        n_timepoints = int(n_timepoints)
        x_padded = np.zeros((n_channels, n_timepoints + 2 * reach), dtype=float)

        x_padded[:, reach : reach + n_timepoints] = x
        x_padded[:, :reach] = np.expand_dims(x[:, 0], axis=-1)
        x_padded[:, reach + n_timepoints :] = np.expand_dims(x[:, -1], axis=-1)

    elif x.ndim == 3:
        n_timepoints = x.shape[2]
        n_channels = x.shape[1]
        n_cases = x.shape[0]
        new_n_timepoints = int(n_timepoints + 2 * reach)

        x_padded = np.zeros(shape=(n_cases, n_channels, new_n_timepoints), dtype=float)

        x_padded[:, :, reach : reach + n_timepoints] = x
        x_padded[:, :, :reach] = np.expand_dims(x[:, :, 0], axis=-1)
        x_padded[:, :, reach + n_timepoints :] = np.expand_dims(x[:, :, -1], axis=-1)
    else:
        raise ValueError("x must be either 2D or 3D array.")

    return x_padded


@njit(cache=True, fastmath=True)
def _identity_descriptor(x: np.ndarray) -> np.ndarray:
    """Return the identity function of the given 1-D subsequence.

    Parameters
    ----------
    x : np.ndarray, shape (n_channels, n_timepoints)
    """
    return np.copy(x)


@njit(cache=True, fastmath=True)
def _transform_subsequences(
    x: np.ndarray, descriptor: str = "identity", reach: int = 30
) -> np.ndarray:
    """Decompose the series into sub-sequences.

    It applies a transformation over each sub-sequence

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    descriptor : str, default=None (if None then identity is used).
        Defines which transformation is applied on the sub-sequences.
        Valid descriptors are: ['identity']

        Identity is simply a copying mechanism of the sub-sequence,
        no transformations are done.
        For now no other descriptors are implemented.

        If not specified then identity is used.
    reach : int, default=30.
        Length of the sub-sequences.

    Returns
    -------
    out_mts : np.ndarray, shape = (new_n_channels, n_timepoints+reach*2).
        The output multivariate time series.
    """
    if descriptor == "identity":
        descriptor_function = _identity_descriptor
    else:
        raise ValueError("Descriptor invalid. Descriptor must be 'identity'.")

    sliding_window = reach * 2 + 1
    sliding_window = int(sliding_window)

    # get the output dimension of the subsequence transofrmation s
    dim_desc = descriptor_function(x[0, 0 : 0 + sliding_window]).shape[0]
    # dim_desc = 7

    n_channels = x.shape[0]
    n_timepoints = x.shape[1] - 2 * reach

    # define the output MTS which has the same
    out_mts = np.zeros(
        (n_channels * dim_desc, n_timepoints),
    )

    # loop through each data point
    for i in range(n_timepoints):
        # loop through each dimension of the MTS
        for j in range(n_channels):
            val = descriptor_function(x[j, i : i + sliding_window])
            out_mts[j * dim_desc : (j + 1) * dim_desc, i] = val
    return out_mts


@njit(cache=True, fastmath=True)
def shape_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 30,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> float:
    """Compute the ShapeDTW distance function between two series x and y.

    The ShapeDTW distance measure was proposed in [1] and used for time series
    averaging in [2] as well.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        Defines which transformation is applied on the sub-sequences.
        Valid descriptors are: ['identity']

        Identity is simply a copying mechanism of the sub-sequence,
        no transformations are done.
        For now no other descriptors are implemented.

        If not specified then identity is used.
    reach : int, default=30.
        Length of the sub-sequences to consider.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    transformation_precomputed : bool, default = False
        To choose if the transformation of the sub-sequences is pre-computed or not.
    transformed_x : np.ndarray, default = None
        The transformation of x, ignored if transformation_precomputed is False.
    transformed_y : np.ndarray, default = None
        The transformation of y, ignored if transformation_precomputed is False.

    Returns
    -------
    float
        ShapeDTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    [1] Zhao, Jiaping, and Laurent Itti. "shapedtw: Shape dynamic time warping."
        Pattern Recognition 74 (2018): 171-184.
    [2] Ali Ismail-Fawaz, Hassan Ismail Fawaz, François Petitjean, Maxime Devanne,
        Jonathan Weber, Stefano Berretti, Geoffrey I. Webb and Germain Forestier.
        "ShapeDBA: Generating Effective Time Series Prototypes using ShapeDTW
        Barycenter Averaging" ECML/PKDD Workshop on Advanced Analytics and
        Learning on Temporal Data, Turin, Italy, 2023.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))

        x_pad = _pad_ts_edges(x=_x, reach=reach)
        y_pad = _pad_ts_edges(x=_y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )

        return _shape_dtw_distance(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
        )
    if x.ndim == 2 and y.ndim == 2:
        x_pad = _pad_ts_edges(x=x, reach=reach)
        y_pad = _pad_ts_edges(x=y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )

        return _shape_dtw_distance(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
        )

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _shape_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    descriptor: str = "identity",
    reach: int = 30,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> float:
    # for compilation purposes
    if transformed_x is None:
        transformed_x = x
    if transformed_y is None:
        transformed_y = y

    if not transformation_precomputed:
        new_x = _transform_subsequences(x=x, descriptor=descriptor, reach=reach)
        new_y = _transform_subsequences(x=y, descriptor=descriptor, reach=reach)

        shape_dtw_cost_mat = _dtw_cost_matrix(
            x=new_x, y=new_y, bounding_matrix=bounding_matrix
        )
    else:
        shape_dtw_cost_mat = _dtw_cost_matrix(
            x=transformed_x, y=transformed_y, bounding_matrix=bounding_matrix
        )

    return _get_shape_dtw_distance_from_cost_mat(
        x=x, y=y, reach=reach, shape_dtw_cost_mat=shape_dtw_cost_mat
    )


@njit(cache=True, fastmath=True)
def _get_shape_dtw_distance_from_cost_mat(
    x: np.ndarray, y: np.ndarray, reach: int, shape_dtw_cost_mat: np.ndarray
) -> float:
    i = shape_dtw_cost_mat.shape[0] - 1
    j = shape_dtw_cost_mat.shape[1] - 1

    shapedtw_dist = 0

    while i >= 0 and j >= 0:
        shapedtw_dist += _univariate_squared_distance(x[:, reach + i], y[:, reach + j])

        a = shape_dtw_cost_mat[i - 1, j - 1]
        b = shape_dtw_cost_mat[i, j - 1]
        c = shape_dtw_cost_mat[i - 1, j]
        if a < b and a < c:
            i -= 1
            j -= 1
        elif b < c:
            j -= 1
        else:
            i -= 1

    return shapedtw_dist


@njit(cache=True, fastmath=True)
def shape_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 30,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the ShapeDTW cost matrix between two series x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        Defines which transformation is applied on the sub-sequences.
        Valid descriptors are: ['identity']

        Identity is simply a copying mechanism of the sub-sequence,
        no transformations are done.
        For now no other descriptors are implemented.

        If not specified then identity is used.
    reach : int, default=30.
        Length of the sub-sequences.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    transformation_precomputed : bool, default = False
        To choose if the transformation of the sub-sequences is pre-computed or not.
    transformed_x : np.ndarray, default = None
        The transformation of x, ignored if transformation_precomputed is False.
    transformed_y : np.ndarray, default = None
        The transformation of y, ignored if transformation_precomputed is False.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        shapedtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))

        x_pad = _pad_ts_edges(x=_x, reach=reach)
        y_pad = _pad_ts_edges(x=_y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )

        return _shape_dtw_cost_matrix(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
        )
    if x.ndim == 2 and y.ndim == 2:
        x_pad = _pad_ts_edges(x=x, reach=reach)
        y_pad = _pad_ts_edges(x=y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )

        return _shape_dtw_cost_matrix(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
        )

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _shape_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    descriptor: str = "identity",
    reach: int = 30,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    # for compilation purposes
    if transformed_x is None:
        transformed_x = x
    if transformed_y is None:
        transformed_y = y

    if not transformation_precomputed:
        new_x = _transform_subsequences(x=x, descriptor=descriptor, reach=reach)
        new_y = _transform_subsequences(x=y, descriptor=descriptor, reach=reach)

        shapedtw_cost_mat = _dtw_cost_matrix(
            x=new_x, y=new_y, bounding_matrix=bounding_matrix
        )
    else:
        shapedtw_cost_mat = _dtw_cost_matrix(
            x=transformed_x, y=transformed_y, bounding_matrix=bounding_matrix
        )

    return shapedtw_cost_mat


@njit(cache=True, fastmath=True)
def shape_dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 30,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the ShapeDTW alignment path between two series x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        Defines which transformation is applied on the sub-sequences.
        Valid descriptors are: ['identity']

        Identity is simply a copying mechanism of the sub-sequence,
        no transformations are done.
        For now no other descriptors are implemented.

        If not specified then identity is used.
    reach : int, default=30.
        Length of the sub-sequences.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    transformation_precomputed : bool, default = False
        To choose if the transformation of the sub-sequences is pre-computed or not.
    transformed_x : np.ndarray, default = None
        The transformation of x, ignored if transformation_precomputed is False.
    transformed_y : np.ndarray, default = None
        The transformation of y, ignored if transformation_precomputed is False.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    cost_matrix = shape_dtw_cost_matrix(
        x=x,
        y=y,
        window=window,
        descriptor=descriptor,
        reach=reach,
        itakura_max_slope=itakura_max_slope,
        transformation_precomputed=transformation_precomputed,
        transformed_x=transformed_x,
        transformed_y=transformed_y,
    )

    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))

        x_pad = _pad_ts_edges(x=_x, reach=reach)
        y_pad = _pad_ts_edges(x=_y, reach=reach)
    elif x.ndim == 2 and y.ndim == 2:
        x_pad = _pad_ts_edges(x=x, reach=reach)
        y_pad = _pad_ts_edges(x=y, reach=reach)
    else:
        raise ValueError("x and y must be 1D or 2D")

    shapedtw_dist = _get_shape_dtw_distance_from_cost_mat(
        x=x_pad, y=y_pad, reach=reach, shape_dtw_cost_mat=cost_matrix
    )

    return (compute_min_return_path(cost_matrix), shapedtw_dist)


@njit(cache=True, fastmath=True)
def shape_dtw_pairwise_distance(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 30,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the ShapeDTW pairwise distance among a set of series.

    Parameters
    ----------
    X : np.ndarray
        A set of time series, either univariate, shape ``(n_cases, n_timepoints,)``,
        or multivariate, shape ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    descriptor : str, default=None (if None then identity is used).
        Defines which transformation is applied on the sub-sequences.
        Valid descriptors are: ['identity']

        Identity is simply a copying mechanism of the sub-sequence,
        no transformations are done.
        For now no other descriptors are implemented.

        If not specified then identity is used.
    reach : int, default=30.
        Length of the sub-sequences.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    transformation_precomputed : bool, default = False
        To choose if the transformation of the sub-sequences is pre-computed or not.
    transformed_x : np.ndarray, default = None
        The transformation of x, ignored if transformation_precomputed is False.
    transformed_y : np.ndarray, default = None
        The transformation of y, ignored if transformation_precomputed is False.

    Returns
    -------
    np.ndarray
        ShapeDTW pairwise matrix between the instances of X of shape
        ``(n_cases, n_cases)`` or between X and y of shape ``(n_cases,
        n_cases)``.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import shape_dtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> shape_dtw_pairwise_distance(X)
    array([[  0.,  27., 108.],
           [ 27.,   0.,  27.],
           [108.,  27.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> shape_dtw_pairwise_distance(X, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> shape_dtw_pairwise_distance(X, y_univariate)
    array([[300.],
           [147.],
           [ 48.]])
    """
    if y is None:
        if X.ndim == 3:
            X_pad = _pad_ts_edges(x=X, reach=reach)
            return _shape_dtw_pairwise_distance(
                X=X_pad,
                window=window,
                descriptor=descriptor,
                reach=reach,
                itakura_max_slope=itakura_max_slope,
                transformation_precomputed=transformation_precomputed,
                transformed_x=transformed_x,
                transformed_y=transformed_y,
            )
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            X_pad = _pad_ts_edges(x=_X, reach=reach)
            return _shape_dtw_pairwise_distance(
                X=X_pad,
                window=window,
                descriptor=descriptor,
                reach=reach,
                itakura_max_slope=itakura_max_slope,
                transformation_precomputed=transformation_precomputed,
                transformed_x=transformed_x,
                transformed_y=transformed_y,
            )
        raise ValueError("X must be 2D or 3D arrays")
    else:
        _X, _y = reshape_pairwise_to_multiple(x=X, y=y)
        X_pad = _pad_ts_edges(x=_X, reach=reach)
        y_pad = _pad_ts_edges(x=_y, reach=reach)

        return _shape_dtw_pairwise_distance(
            X=X_pad,
            y=y_pad,
            window=window,
            descriptor=descriptor,
            reach=reach,
            itakura_max_slope=itakura_max_slope,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
        )


@njit(cache=True, fastmath=True)
def _shape_dtw_pairwise_distance(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 30,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    if y is None:
        y = np.copy(X)

    distances = np.zeros(shape=(len(X), len(y)))
    bounding_matrix = create_bounding_matrix(
        X.shape[2] - 2 * reach, y.shape[2] - 2 * reach, window, itakura_max_slope
    )

    for i in range(len(X)):
        for j in range(len(y)):
            distances[i, j] = _shape_dtw_distance(
                x=X[i],
                y=y[j],
                descriptor=descriptor,
                reach=reach,
                bounding_matrix=bounding_matrix,
                transformation_precomputed=transformation_precomputed,
                transformed_x=transformed_x,
                transformed_y=transformed_y,
            )

    return distances
