"""Series testing utils."""

__maintainer__ = []
__all__ = ["make_series", "make_forecasting_problem"]

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


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
