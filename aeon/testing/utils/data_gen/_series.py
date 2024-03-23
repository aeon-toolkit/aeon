"""Series testing utils."""

__maintainer__ = []
__all__ = ["make_series", "make_forecasting_problem"]

from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def make_2d_numpy_series(
    n_channels: int = 2,
    n_timepoints: int = 8,
    random_state: Union[int, None] = None,
    axis: int = 1,
) -> np.ndarray:
    """Randomly generate 2D series for testing.

    Parameters
    ----------
    n_channels : int
        The number of channels to generate.
    n_timepoints : int
        The length of the series to generate.
    random_state : int or None
        Seed for random number generation.
    axis : int, default = 1
        Axis along which to segment if passed a multivariate series (2D input).
        If axis is 0, it is assumed each column is a time series and each row
        is a timepoint. i.e. the shape of the data is
        ``(n_timepoints,n_channels)``. ``axis == 1`` indicates the time series
        are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.

    Returns
    -------
    X : np.ndarray
        Randomly generated 2D series.

    Examples
    --------
    >>> data = make_2d_numpy_series(n_channels=2, n_timepoints=6,
    ... random_state=0)
    >>> print(data)
    [[0.5488135  0.60276338 0.4236548  0.43758721 0.96366276 0.79172504]
     [0.71518937 0.54488318 0.64589411 0.891773   0.38344152 0.52889492]]
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(size=(n_timepoints, n_channels))
    if axis == 1:
        X = X.T
    return X


def make_1d_numpy_series(
    n_timepoints: int = 8,
    random_state: Union[int, None] = None,
) -> np.ndarray:
    """Randomly generate 1D series for testing.

    Parameters
    ----------
    n_timepoints : int
        The length of the series to generate.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 1D series.

    Examples
    --------
    >>> data = make_1d_numpy_series(n_timepoints=6, random_state=0)
    >>> print(data)
    [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548  0.64589411]
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(size=n_timepoints)
    return X


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
        start = "2000-01"
        freq = "M"
        return pd.period_range(start=start, periods=n_timepoints, freq=freq)

    elif index_type == "datetime" or index_type is None:
        start = "2000-01-01"
        freq = "D"
        return pd.date_range(start=start, periods=n_timepoints, freq=freq)

    elif index_type == "range":
        start = 3  # check non-zero based indices
        return pd.RangeIndex(start=start, stop=start + n_timepoints)

    elif index_type == "int":
        start = 3
        return pd.Index(np.arange(start, start + n_timepoints), dtype=int)

    else:
        raise ValueError(f"index_class: {index_type} is not supported")


def make_forecasting_problem(
    n_timepoints: int = 50,
    all_positive: bool = True,
    index_type=None,
    make_X: bool = False,
    n_columns: int = 1,
    random_state=None,
):
    """Return test data for forecasting tests.

    Parameters
    ----------
    n_timepoints : int, default = 50
        Num of timepoints in series.
    all_positive : bool, default = True
        Only positive values or not.
    index_type : pd.PeriodIndex or None, default = None
        pandas Index type to use.
    make_X : bool, default = False
        Should X data also be returned.
    n_columns : int, default = 1
        Number of columns of y.
    random_state : inst, str, float, default=None
        Set seed of random state

    Returns
    -------
    pd.Series
        generated series if not make_X
    (pd.Series, pd.DataFrame)
        (pd.Series, pd.DataFrame) if make_X
    """
    y = make_series(
        n_timepoints=n_timepoints,
        n_columns=n_columns,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )

    if not make_X:
        return y

    X = make_series(
        n_timepoints=n_timepoints,
        n_columns=2,
        all_positive=all_positive,
        index_type=index_type,
        random_state=random_state,
    )
    X.index = y.index
    return y, X
