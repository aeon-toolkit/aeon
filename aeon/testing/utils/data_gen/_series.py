"""Series testing utils."""

__author__ = ["mloning"]
__all__ = []

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def make_series(
    n_timepoints=50,
    n_columns=1,
    all_positive=True,
    index_type=None,
    return_numpy=False,
    random_state=None,
    add_nan=False,
):
    """Generate univariate or multivariate time series."""
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
    n_timepoints=50,
    all_positive=True,
    index_type=None,
    make_X=False,
    n_columns=1,
    random_state=None,
):
    """Return test data for forecasting tests.

    Parameters
    ----------
    n_timepoints : int, optional
        Lenght of data, by default 50
    all_positive : bool, optional
        Only positive values or not, by default True
    index_type : e.g. pd.PeriodIndex, optional
        pandas Index type, by default None
    make_X : bool, optional
        Should X data also be returned, by default False
    n_columns : int, optional
        Number of columns of y, by default 1
    random_state : inst, str, float, optional
        Set seed of random state, by default None

    Returns
    -------
    ps.Series, pd.DataFrame
        y, if not make_X
        y, X if make_X
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
