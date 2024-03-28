from typing import List, Tuple, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape two collections of time series for pairwise distance computation.

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
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return x, _y
        if y.ndim == 3 and x.ndim == 2:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
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


def _convert_to_list(
    x: Union[np.ndarray, List[np.ndarray]], name: str = "X"
) -> NumbaList[np.ndarray]:
    """Convert input collections to a list of arrays for pairwise distance calculation.

    Takes a single or multiple time series and converts them to a list of 2D arrays. If
    the input is a single time series, it is reshaped to a 2D array as the sole element
    of a list. If the input is a 2D array of shape (n_cases, n_timepoints), it is
    reshaped to a list of n_cases 1D arrays with n_timepoints points. A 3D array is
    converted to a list with n_cases 2D arrays of shape (n_channels, n_timepoints).
    Lists of 1D arrays are converted to lists of 2D arrays.

    Parameters
    ----------
    x : Union[np.ndarray, List[np.ndarray]]
        One or more time series of shape (n_cases, n_channels, n_timepoints) or
        (n_cases, n_timepoints) or (n_timepoints,).
    name : str, optional
        Name of the variable to be converted for error handling, by default "X".

    Returns
    -------
    NumbaList[np.ndarray]
        Numba typedList of 2D arrays with shape (n_channels, n_timepoints) of length
        n_cases.

    Raises
    ------
    ValueError
        If x is not a 1D, 2D or 3D array or a list of 1D or 2D arrays.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return NumbaList(x)
        elif x.ndim == 2:
            return NumbaList(x.reshape(x.shape[0], 1, x.shape[1]))
        elif x.ndim == 1:
            return NumbaList(x.reshape(1, 1, x.shape[0]))
        else:
            raise ValueError(f"{name} must be 1D, 2D or 3D")
    elif isinstance(x, (List, NumbaList)):
        x_new = NumbaList()
        for i in range(len(x)):
            if x[i].ndim == 2:
                x_new.append(x[i])
            elif x[i].ndim == 1:
                x_new.append(x[i].reshape((1, x[i].shape[0])))
            else:
                raise ValueError(f"{name} must include only 1D or 2D arrays")
        return x_new
    else:
        raise ValueError(f"{name} must be either np.ndarray or List[np.ndarray]")
