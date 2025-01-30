"""Test series module."""

__maintainer__ = ["TonyBagnall"]

from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest

from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_series,
    make_example_pandas_series,
)
from aeon.utils.validation.series import (
    check_series,
    is_hierarchical,
    is_single_series,
    is_univariate_series,
)


def test_is_univariate_series():
    """Test is_univariate_series."""
    assert not is_univariate_series(None)
    assert is_univariate_series(make_example_pandas_series())
    assert is_univariate_series(make_example_1d_numpy())
    assert is_univariate_series(make_example_dataframe_series(n_channels=1))
    assert not is_univariate_series(make_example_dataframe_series(n_channels=2))
    assert not is_univariate_series(make_example_2d_numpy_series())
    assert not is_univariate_series(make_example_3d_numpy())
    assert not is_univariate_series(make_example_3d_numpy_list())


def test_is_hierarchical():
    """Test is_hierarchical."""
    assert is_hierarchical(_make_hierarchical())
    assert not is_hierarchical(make_example_dataframe_series(n_channels=2))
    assert not is_hierarchical(make_example_1d_numpy())
    assert not is_hierarchical(make_example_2d_numpy_series())
    assert not is_hierarchical(make_example_3d_numpy_list())


def test_is_single_series():
    """Test is_single_series."""
    assert not is_single_series(None)
    assert is_single_series(make_example_pandas_series())
    assert is_single_series(make_example_1d_numpy())
    assert is_univariate_series(make_example_dataframe_series(n_channels=1))
    assert is_single_series(make_example_dataframe_series(n_channels=2))
    assert is_single_series(make_example_2d_numpy_series())
    assert not is_univariate_series(make_example_3d_numpy())
    assert not is_univariate_series(make_example_3d_numpy_list())


def test_check_series():
    """Test check_series."""
    assert isinstance(make_example_pandas_series(), pd.Series)
    assert isinstance(make_example_1d_numpy(), np.ndarray)
    assert isinstance(make_example_dataframe_series(n_channels=2), pd.DataFrame)
    with pytest.raises(ValueError, match="Input type of y should be one "):
        check_series(None)
    # check


def _make_hierarchical(
    hierarchy_levels: tuple = (2, 4),
    max_timepoints: int = 12,
    min_timepoints: int = 12,
    same_cutoff: bool = True,
    n_columns: int = 1,
    all_positive: bool = True,
    index_type: Optional[str] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    add_nan: bool = False,
) -> pd.DataFrame:
    """Generate hierarchical multiindex type for testing.

    Parameters
    ----------
    hierarchy_levels : Tuple, optional
        the number of groups at each hierarchy level, by default (2, 4)
    max_timepoints : int, optional
        maximum time points a series can have, by default 12
    min_timepoints : int, optional
        minimum time points a seires can have, by default 12
    same_cutoff : bool, optional
        If it's True all series will end at the same date, by default True
    n_columns : int, optional
        number of columns in the output dataframe, by default 1
    all_positive : bool, optional
        If True the time series will be , by default True
    index_type : str, optional
        type of index, by default None
        Supported types are "period", "datetime", "range" or "int".
        If it's not provided, "datetime" is selected.
    random_state : int, np.random.RandomState or None
        Controls the randomness of the estimator, by default None
    add_nan : bool, optional
        If it's true the series will contain NaNs, by default False

    Returns
    -------
    pd.DataFrame
        hierarchical dataframe
    """
    from itertools import product

    from sklearn.utils import check_random_state

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

    levels = [
        [f"h{i}_{j}" for j in range(hierarchy_levels[i])]
        for i in range(len(hierarchy_levels))
    ]
    level_names = [f"h{i}" for i in range(len(hierarchy_levels))]
    rng = check_random_state(random_state)
    if min_timepoints == max_timepoints:
        time_index = _make_index(max_timepoints, index_type)
        index = pd.MultiIndex.from_product(
            levels + [time_index], names=level_names + ["time"]
        )
    else:
        df_list = []
        for levels_tuple in product(*levels):
            n_timepoints = rng.randint(low=min_timepoints, high=max_timepoints)
            if same_cutoff:
                time_index = _make_index(max_timepoints, index_type)[-n_timepoints:]
            else:
                time_index = _make_index(n_timepoints, index_type)
            d = dict(zip(level_names, levels_tuple))
            d["time"] = time_index
            df_list.append(pd.DataFrame(d))
        index = pd.MultiIndex.from_frame(
            pd.concat(df_list), names=level_names + ["time"]
        )

    total_time_points = len(index)
    data = rng.normal(size=(total_time_points, n_columns))
    if add_nan:
        # add some nan values
        data[int(len(data) / 2)] = np.nan
        data[0] = np.nan
        data[-1] = np.nan
    if all_positive:
        data -= np.min(data, axis=0) - 1
    df = pd.DataFrame(
        data=data, index=index, columns=[f"c{i}" for i in range(n_columns)]
    )

    return df
