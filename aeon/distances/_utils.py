from typing import Tuple

import numpy as np
from numba import njit
from sklearn.utils import check_random_state


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple(
    x: np.ndarray, y: np.ndarray, ensure_equal_dims: bool = False
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
    ensure_equal_dims : bool, default=False
        If True, x and y must have the same number of dimensions; otherwise an error is
        raised. If False, the function reshapes both arrays to 3D arrays.

    Returns
    -------
    np.ndarray,
        Reshaped x.
    np.ndarray
        Reshaped y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D or 3D arrays, and if `ensure_equal_dims` is True and
        x and y do not have the same number of dimensions.
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
        if x.ndim == 2 and y.ndim == 1:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        if y.ndim == 2 and x.ndim == 1:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        if ensure_equal_dims:
            raise ValueError("x and y must have the same number of dimensions")
        else:
            if x.ndim == 3 and y.ndim == 2:
                _y = y.reshape((1, y.shape[0], y.shape[1]))
                return x, _y
            if y.ndim == 3 and x.ndim == 2:
                _x = x.reshape((1, x.shape[0], x.shape[1]))
                return _x, y
            raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def _reshape_pairwise_single(
    x: np.ndarray, y: np.ndarray, ensure_equal_dims: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape two time series for pairwise distance computation.

    If x and y are 1D arrays (univariate time series), they are reshaped to (1, n) and
    (1, m) keeping their individual length. If x and y are already 2D arrays
    (multivariate time series), they keep their shape. If `ensure_equal_dims` is True,
    both x and y need to have the same number of dimensions. If False, the function
    reshapes them to 2D arrays with the 1D array having a shape (1, n) as above.

    Parameters
    ----------
    x : np.ndarray
        One time series of shape (n_timepoints,) or (n_channels, n_timepoints).
    y : np.ndarray
        One time series of shape (m_timepoints,) or (m_channels, m_timepoints).
    ensure_equal_dims : bool, default=False
        If True, x and y must have the same number of dimensions; otherwise an error is
        raised. If False, the function reshapes the 1D array to (1, n).

    Returns
    -------
    np.ndarray,
        Reshaped x.
    np.ndarray
        Reshaped y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays, and if `ensure_equal_dims` is True and x and
        y do not have the same number of dimensions.
    """
    if x.ndim == y.ndim:
        if y.ndim == 2 and x.ndim == 2:
            return x, y
        if y.ndim == 1 and x.ndim == 1:
            _x = x.reshape((1, x.shape[0]))
            _y = y.reshape((1, y.shape[0]))
            return _x, _y
        raise ValueError("x and y must be 1D or 2D arrays")

    if not ensure_equal_dims:
        if x.ndim == 2 and y.ndim == 1:
            _y = y.reshape((1, y.shape[0]))
            return x, _y
        if y.ndim == 2 and x.ndim == 1:
            _x = x.reshape((1, x.shape[0]))
            return _x, y
        raise ValueError("x and y must be 1D or 2D arrays")

    raise ValueError("x and y must have the same number of dimensions")


def _create_test_distance_numpy(
    n_instance: int,
    n_channels: int = None,
    n_timepoints: int = None,
    random_state: int = 1,
):
    """Create a test numpy distance.

    Parameters
    ----------
    n_instance: int
        Number of instances to create.
    n_channels: int
        Number of channels to create.
    n_timepoints: int, default=None
        Number of timepoints to create in each channel.
    random_state: int, default=1
        Random state to initialise with.

    Returns
    -------
    np.ndarray 2D or 3D numpy
        Numpy array of shape specific. If 1 instance then 2D array returned,
        if > 1 instance then 3D array returned.
    """
    rng = check_random_state(random_state)
    # Generate data as 3d numpy array
    if n_timepoints is None and n_channels is None:
        return rng.normal(scale=0.5, size=(1, n_instance))
    if n_timepoints is None:
        return rng.normal(scale=0.5, size=(n_instance, n_channels))
    return rng.normal(scale=0.5, size=(n_instance, n_channels, n_timepoints))


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
