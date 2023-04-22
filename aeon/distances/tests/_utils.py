# -*- coding: utf-8 -*-
import os
import time
from typing import Callable

import numpy as np

from aeon.datatypes import convert_to
from aeon.utils._testing.panel import _make_panel_X
from aeon.utils._testing.series import _make_series


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
    num_dims = 3
    if n_timepoints is None:
        n_timepoints = 1
        num_dims -= 1
    if n_columns is None:
        n_columns = 1
        num_dims -= 1

    df = _create_test_distances(
        n_instance=n_instance,
        n_columns=n_columns,
        n_timepoints=n_timepoints,
        random_state=random_state,
    )
    if num_dims == 3:
        return convert_to(df, to_type="numpy3D")
    elif num_dims == 2:
        return convert_to(df, to_type="numpy3D")[:, :, 0]
    else:
        return convert_to(df, to_type="numpy3D")[:, 0, 0]


def _create_test_distances(n_instance, n_columns, n_timepoints, random_state=1):
    if n_instance > 1:
        return _make_panel_X(
            n_instances=n_instance,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            random_state=random_state,
        )
    else:
        return _make_series(n_timepoints, n_columns, random_state=random_state)


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


def debug_generated_jit_distance_function(func):
    """Check if numba generated_jit function is active for a test.

    When using generated_jit function and numba is active this creates a function that
    returns the value of the returned callable. However, when numba isnt active
    this isnt resolved so you just get the resolved callable back. This means that
    these functions cant be tested without numba being active. This function wraps
    generated_jit functions when numba isnt active so they can be used even when
    numba isnt active.

    Parameters
    ----------
    func: Callable
        The function that could be a numba generated_jit.

    Returns
    -------
    Callable
        The function that is wrapped if numba isnt active.
    """
    if "NUMBA_DISABLE_JIT" in os.environ and os.environ["NUMBA_DISABLE_JIT"] == "1":

        def dist_callable(x, y):
            inner_callable = func(x, y)
            return inner_callable(x, y)

        return dist_callable
    return func
