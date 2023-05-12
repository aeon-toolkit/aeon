# -*- coding: utf-8 -*-
import time
from typing import Callable

import numpy as np
from sklearn.utils.validation import check_random_state


def create_test_distance_numpy(
    n_instance: int,
    n_columns: int = None,
    n_timepoints: int = None,
    random_state: int = 1,
):
    """Create a test numpy distance.

    Parameters
    ----------
    n_instance: int
        Number of instances to create.
    n_columns: int
        Number of columns to create.
    n_timepoints: int, defaults = None
        Number of timepoints to create in each column.
    random_state: int, defaults = 1
        Random state to initialise with.

    Returns
    -------
    np.ndarray 2D or 3D numpy
        Numpy array of shape specific. If 1 instance then 2D array returned,
        if > 1 instance then 3D array returned.
    """
    rng = check_random_state(random_state)
    # Generate data as 3d numpy array
    if n_timepoints is None and n_columns is None:
        return rng.normal(scale=0.5, size=(1, n_instance))
    if n_timepoints is None:
        return rng.normal(scale=0.5, size=(n_instance, n_columns))
    return rng.normal(scale=0.5, size=(n_instance, n_columns, n_timepoints))


def _time_distance(callable: Callable, average: int = 30, **kwargs):
    for _ in range(3):
        callable(**kwargs)

    total = 0
    for _ in range(average):
        start = time.time()
        callable(**kwargs)
        total += time.time() - start

    return total / average


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
