"""Utility functions for generating series testing data."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "make_example_1d_numpy",
    "make_example_2d_numpy_series",
    "make_example_pandas_series",
    "make_example_dataframe_series",
]

from typing import Union

import numpy as np
import pandas as pd


def make_example_1d_numpy(
    n_timepoints: int = 12,
    random_state: Union[int, None] = None,
) -> np.ndarray:
    """Randomly generate 1D numpy X.

    Generates data in 1D 'np.ndarray' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    random_state : int or None, default=None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 1D data.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_1d_numpy
    >>> data = make_example_1d_numpy(
    ...     n_timepoints=8,
    ...     random_state=0,
    ... )
    >>> print(data)
    [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548  0.64589411
     0.43758721 0.891773  ]
    """
    rng = np.random.RandomState(random_state)
    return rng.uniform(size=(n_timepoints,))


def make_example_2d_numpy_series(
    n_timepoints: int = 12,
    n_channels: int = 1,
    random_state: Union[int, None] = None,
    axis: int = 1,
) -> np.ndarray:
    """Randomly generate 2D numpy X.

    Generates data in 2D 'np.ndarray' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    n_channels : int, default=1
        The number of series channels to generate.
    random_state : int or None, default=None
        Seed for random number generation.
    axis : int, default=1
        The axis to for the series timepoints. If 1, returns the shape
        (n_channels, n_timepoints). If 0, returns the shape (n_timepoints, n_channels).

    Returns
    -------
    X : np.ndarray
        Randomly generated 2D data.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_2d_numpy_series
    >>> data = make_example_2d_numpy_series(
    ...     n_timepoints=6,
    ...     n_channels=2,
    ...     random_state=0,
    ...     axis=0,
    ... )
    >>> print(data)
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]
     [0.4236548  0.64589411]
     [0.43758721 0.891773  ]
     [0.96366276 0.38344152]
     [0.79172504 0.52889492]]
    """
    rng = np.random.RandomState(random_state)
    if axis == 1:
        return rng.uniform(size=(n_channels, n_timepoints))
    elif axis == 0:
        return rng.uniform(size=(n_timepoints, n_channels))
    else:
        raise ValueError(f"axis: {axis} is not supported, please use 0 or 1.")


def make_example_pandas_series(
    n_timepoints: int = 12,
    index_type=None,
    random_state: Union[int, None] = None,
) -> pd.Series:
    """Randomly generate pandas Series X.

    Generates data in 'pd.Series' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    index_type : str or None, default=None
        pandas Index type to use. One of ["period", "datetime", "range", "int"].
        If None, uses default integer index.
    random_state : int or None, default=None
        Seed for random number generation.

    Returns
    -------
    X : pd.Series
        Randomly generated 1D data.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_pandas_series
    >>> data = make_example_pandas_series(
    ...     n_timepoints=6,
    ...     random_state=0,
    ... )
    >>> print(data)
    0    0.548814
    1    0.715189
    2    0.602763
    3    0.544883
    4    0.423655
    5    0.645894
    dtype: float64
    """
    rng = np.random.RandomState(random_state)
    index = _make_index(n_timepoints, index_type)
    return pd.Series(rng.uniform(size=(n_timepoints,)), index=index)


def make_example_dataframe_series(
    n_timepoints: int = 12,
    n_channels: int = 1,
    index_type=None,
    random_state: Union[int, None] = None,
    axis: int = 1,
) -> pd.DataFrame:
    """Randomly generate pandas DataFrame X.

    Generates data in 'pd.DataFrame' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    n_channels : int, default=1
        The number of series channels to generate.
    index_type : str or None, default=None
        pandas Index type to use. One of ["period", "datetime", "range", "int"].
        If None, uses default integer index.
    random_state : int or None, default=None
        Seed for random number generation.
    axis : int, default=1
        The axis to for the series timepoints. If 1, returns the shape
        (n_channels, n_timepoints). If 0, returns the shape (n_timepoints, n_channels).

    Returns
    -------
    X : pd.DataFrame
        Randomly generated 2D data.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_dataframe_series
    >>> data = make_example_dataframe_series(
    ...     n_timepoints=6,
    ...     n_channels=2,
    ...     random_state=0,
    ...     axis=0,
    ... )
    >>> print(data)
              0         1
    0  0.548814  0.715189
    1  0.602763  0.544883
    2  0.423655  0.645894
    3  0.437587  0.891773
    4  0.963663  0.383442
    5  0.791725  0.528895
    """
    rng = np.random.RandomState(random_state)
    index = _make_index(n_timepoints, index_type)
    if axis == 1:
        return pd.DataFrame(
            rng.uniform(size=(n_channels, n_timepoints)),
            index=np.arange(n_channels),
            columns=index,
        )
    elif axis == 0:
        return pd.DataFrame(
            rng.uniform(size=(n_timepoints, n_channels)),
            columns=np.arange(n_channels),
            index=index,
        )
    else:
        raise ValueError(f"axis: {axis} is not supported, please use 0 or 1.")


def _make_index(n_timepoints, index_type=None):
    """Make indices for unit testing."""
    if index_type == "period":
        return pd.period_range(start="2000-01", periods=n_timepoints, freq="M")
    elif index_type == "datetime":
        return pd.date_range(start="2000-01-01", periods=n_timepoints, freq="D")
    elif index_type == "range":
        return pd.RangeIndex(start=0, stop=n_timepoints)
    elif index_type == "int" or index_type is None:
        return pd.Index(np.arange(0, n_timepoints), dtype=int)
    else:
        raise ValueError(f"index_class: {index_type} is not supported")
