from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape two collection of time series for pairwise distance computation.

    Parameters
    ----------
    x : np.ndarray
        One or more time series of shape (n_cases, n_channels,
        n_timepoints) or
            (n_cases, n_timepoints) or (n_timepoints,).
    y : np.ndarray
        One or more time series of shape (m_cases, m_channels, m_timepoints) or
            (m_cases, m_timepoints) or (m_timepoints,)

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
        raise ValueError("x and y must be 2D or 3D arrays")
