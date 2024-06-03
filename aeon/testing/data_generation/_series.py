"""Series testing utils."""

__maintainer__ = []
__all__ = []

from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


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
    >>> data, labels = make_example_1d_numpy(
    ...     n_timepoints=8,
    ...     random_state=0,
    ... )
    >>> print(data)
    [[[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
      [0.87517442 1.783546   1.92732552 0.76688304 1.58345008 1.05778984]]
    <BLANKLINE>
     [[2.         3.70238655 0.28414423 0.3485172  0.08087359 3.33047938]
      [3.112627   3.48004859 3.91447337 3.19663426 1.84591745 3.12211671]]]
    """
    rng = np.random.RandomState(random_state)
    return rng.uniform(size=(n_timepoints,))


def make_example_2d_numpy_series(
    n_timepoints: int = 12,
    n_channels: int = 1,
    random_state: Union[int, None] = None,
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

    Returns
    -------
    X : np.ndarray
        Randomly generated 2D data.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_2d_numpy_series
    >>> data, labels = make_example_2d_numpy_series(
    ...     n_channels=2,
    ...     n_timepoints=6,
    ...     random_state=0,
    ... )
    >>> print(data)
    [[[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
      [0.87517442 1.783546   1.92732552 0.76688304 1.58345008 1.05778984]]
    <BLANKLINE>
     [[2.         3.70238655 0.28414423 0.3485172  0.08087359 3.33047938]
      [3.112627   3.48004859 3.91447337 3.19663426 1.84591745 3.12211671]]]
    """
    rng = np.random.RandomState(random_state)
    return rng.uniform(size=(n_timepoints, n_channels))


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
    >>> from aeon.testing.data_generation import make_example_2d_numpy_series
    >>> data, labels = make_example_2d_numpy_series(
    ...     n_channels=2,
    ...     n_timepoints=6,
    ...     random_state=0,
    ... )
    >>> print(data)
    [[[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
      [0.87517442 1.783546   1.92732552 0.76688304 1.58345008 1.05778984]]
    <BLANKLINE>
     [[2.         3.70238655 0.28414423 0.3485172  0.08087359 3.33047938]
      [3.112627   3.48004859 3.91447337 3.19663426 1.84591745 3.12211671]]]
    """
    rng = np.random.RandomState(random_state)
    index = _make_index(n_timepoints, index_type)
    return pd.Series(rng.uniform(size=(n_timepoints,)), index=index)


def make_example_dataframe_series(
    n_timepoints: int = 12,
    n_channels: int = 1,
    index_type=None,
    random_state: Union[int, None] = None,
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

    Returns
    -------
    X : pd.DataFrame
        Randomly generated 2D data.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_2d_numpy_series
    >>> data, labels = make_example_2d_numpy_series(
    ...     n_channels=2,
    ...     n_timepoints=6,
    ...     random_state=0,
    ... )
    >>> print(data)
    [[[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
      [0.87517442 1.783546   1.92732552 0.76688304 1.58345008 1.05778984]]
    <BLANKLINE>
     [[2.         3.70238655 0.28414423 0.3485172  0.08087359 3.33047938]
      [3.112627   3.48004859 3.91447337 3.19663426 1.84591745 3.12211671]]]
    """
    rng = np.random.RandomState(random_state)
    index = _make_index(n_timepoints, index_type)
    return pd.DataFrame(rng.uniform(size=(n_timepoints, n_channels)), index=index)


def make_series(
    n_timepoints: int = 50,
    n_columns: int = 1,
    all_positive: bool = True,
    index_type=None,
    return_numpy: bool = False,
    random_state=None,
    add_nan: bool = False,
):
    """Generate univariate or multivariate time series.

    Parameters
    ----------
    n_timepoints : int, default = 50
        Num of timepoints in series.
    n_columns : int, default = 1
        Number of columns of y.
    all_positive : bool, default = True
        Only positive values or not.
    index_type : pd.PeriodIndex or None, default = None
        pandas Index type to use.
    random_state : inst, str, float, default=None
        Set seed of random state
    add_nan : bool, default = False
        Add nan values to the series.

    Returns
    -------
    np.ndarray, pd.Series, pd.DataFrame
        np.ndarray if return_numpy is True
        pd.Series if n_columns == 1
        else pd.DataFrame
    """
    rng = check_random_state(random_state)
    data = rng.normal(size=(n_timepoints, n_columns))
    if add_nan:
        # add some nan values
        data[len(data) // 2] = np.nan
        data[0] = np.nan
        data[-1] = np.nan
    if all_positive:
        data -= np.min(data, axis=0) - 1
    if return_numpy:
        if n_columns == 1:
            data = data.ravel()
        return data
    else:
        index = _make_index(n_timepoints, index_type)
        if n_columns == 1:
            return pd.Series(data.ravel(), index)
        else:
            return pd.DataFrame(data, index)


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
