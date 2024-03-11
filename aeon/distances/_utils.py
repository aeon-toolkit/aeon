from typing import Optional, Tuple

import numpy as np
from numba import njit
from numba.typed import List


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape two collections of time series for pairwise distance computation.

    Parameters
    ----------
    x : np.ndarray
        One or more time series of shape (n_instances, n_channels,
        n_timepoints) or
            (n_instances, n_timepoints) or (n_timepoints,).
    y : np.ndarray
        One or more time series of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,)

    Returns
    -------
    np.ndarray,
        Reshaped x.
    np.ndarray
        Reshaped y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D or 3D arrays.
    """
    if x.ndim == y.ndim:
        if y.ndim == 3 and x.ndim == 3:
            return x, y
        if y.ndim == 2 and x.ndim == 2:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        if y.ndim == 1 and x.ndim == 1:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        if x.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return x, _y
        if y.ndim == 3 and x.ndim == 2:
            _x = x.reshape((1, x.shape[0], x.shape[1]))
            return _x, y
        if x.ndim == 3 and y.ndim == 1:
            _x = x
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        if x.ndim == 1 and y.ndim == 3:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y
            return _x, _y
        if x.ndim == 2 and y.ndim == 1:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        if y.ndim == 2 and x.ndim == 1:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def _reshape_pairwise_single(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape two time series for pairwise distance computation.

    If x and y are 1D arrays (univariate time series), they are reshaped to (1, n) and
    (1, m) keeping their individual length. If x and y are already 2D arrays
    (multivariate time series), they keep their shape.

    Parameters
    ----------
    x : np.ndarray
        One time series of shape (n_timepoints,) or (n_channels, n_timepoints).
    y : np.ndarray
        One time series of shape (m_timepoints,) or (m_channels, m_timepoints).

    Returns
    -------
    np.ndarray,
        Reshaped x.
    np.ndarray
        Reshaped y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if x.ndim == y.ndim:
        if y.ndim == 2 and x.ndim == 2:
            return x, y
        if y.ndim == 1 and x.ndim == 1:
            _x = x.reshape((1, x.shape[0]))
            _y = y.reshape((1, y.shape[0]))
            return _x, _y
        raise ValueError("x and y must be 1D or 2D arrays")
    else:
        if x.ndim == 2 and y.ndim == 1:
            _y = y.reshape((1, y.shape[0]))
            return x, _y
        if y.ndim == 2 and x.ndim == 1:
            _x = x.reshape((1, x.shape[0]))
            return _x, y
        raise ValueError("x and y must be 1D or 2D arrays")


@njit(cache=True, fastmath=True)
def _ensure_equal_dims(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    if x.ndim == 3 and y.ndim == 3 and x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of channels")

    if x.ndim == 2 and y.ndim == 2 and x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of channels")


@njit(cache=True, fastmath=True)
def _ensure_equal_dims_in_list(
    x: List[np.ndarray], y: Optional[List[np.ndarray]] = None
) -> None:
    x_shapes = [a.shape for a in x]
    x_dims = np.array([a.ndim for a in x], dtype=np.int_)

    if y is None:
        if np.any(x_dims == 1):
            # all time series must be univariate
            multivariate_ts = np.array(
                [s[0] != 1 for s in x_shapes if len(s) == 2], dtype=np.bool_
            )
            if np.any(multivariate_ts):
                raise ValueError("x and y must have the same number of channels")
        else:
            # the number of channels must be the same for multivariate time series
            x_n_channels = [s[0] for s in x_shapes]
            if sum(x_n_channels) / len(x_n_channels) != x_n_channels[0]:
                raise ValueError("x and y must have the same number of channels")

    else:
        y_shapes = [a.shape for a in y]
        y_dims = np.array([a.ndim for a in y], dtype=np.int_)
        if np.any(x_dims == 1) or np.any(y_dims == 1):
            # all time series must be univariate
            multivariate_ts = np.array(
                [s[0] != 1 for s in x_shapes if len(s) == 2]
                + [s[0] != 1 for s in y_shapes if len(s) == 2],
                dtype=np.bool_,
            )
            if np.any(multivariate_ts):
                raise ValueError("x and y must have the same number of channels")
        else:
            # the number of channels must be the same for multivariate time series
            n_channels = [s[0] for s in x_shapes] + [s[0] for s in y_shapes]
            if sum(n_channels) / len(n_channels) != n_channels[0]:
                raise ValueError("x and y must have the same number of channels")


def _make_3d_series(x: np.ndarray) -> np.ndarray:
    """Check a series being passed into pairwise is 3d.

    Pairwise assumes it has been passed two sets of series, if passed a single
    series this function reshapes.

    If given a 1d array the time series is reshaped to (m, 1, 1). This is so when
    looped over x[i] = (1, m).

    If given a 2d array then the time series is reshaped to (d, 1, m). The dimensions
    are put to the start so the ts can be looped through correctly. When looped over
    the time series x[i] = (d, m).

    Parameters
    ----------
    x: np.ndarray, 2d or 3d

    Returns
    -------
    np.ndarray, 3d
    """
    num_dims = x.ndim
    if num_dims == 1:
        shape = x.shape
        _x = np.reshape(x, (1, 1, shape[0]))
    elif num_dims == 2:
        shape = x.shape
        _x = np.reshape(x, (shape[0], 1, shape[1]))
    elif num_dims > 3:
        raise ValueError(
            "The matrix provided has more than 3 dimensions. This is not"
            "supported. Please provide a matrix with less than "
            "3 dimensions"
        )
    else:
        _x = x
    return _x
