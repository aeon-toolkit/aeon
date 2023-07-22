# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Data generators."""

__author__ = ["MatthewMiddlehurst", "TonyBagnall"]
__all__ = [
    "make_example_3d_numpy",
    "make_example_2d_numpy",
    "make_example_long_table",
    "make_example_multi_index_dataframe",
]

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from aeon.utils._testing.collection import make_2d_test_data, make_3d_test_data


def make_example_3d_numpy(
    n_cases: int = 10,
    n_channels: int = 1,
    n_timepoints: int = 12,
    return_y: bool = False,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Randomly generate 3D X and y data.

    Will ensure there is at least one sample per label if a classification
    label is being returned (return_y=True and regression_target=False).

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    n_channels : int
        The number of series channels to generate.
    n_timepoints : int
        The number of features/series length to generate.
    return_y : bool
        If True, return the labels as well as the data.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a scalar float, otherwise an int.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 3D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from aeon.datasets import make_example_3d_numpy
    >>> data, labels = make_example_3d_numpy(n_cases=2, n_channels=2, n_timepoints=6,
    ...                                      return_y=True, n_labels=2, random_state=0)
    >>> print(data)
    [[[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
      [0.87517442 1.783546   1.92732552 0.76688304 1.58345008 1.05778984]]
    <BLANKLINE>
     [[2.         3.70238655 0.28414423 0.3485172  0.08087359 3.33047938]
      [3.112627   3.48004859 3.91447337 3.19663426 1.84591745 3.12211671]]]
    >>> print(labels)
    [0 1]
    """
    X, y = make_3d_test_data(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_labels=n_labels,
        regression_target=regression_target,
        random_state=random_state,
    )
    if return_y:
        return X, y
    return X


def make_example_2d_numpy(
    n_cases: int = 10,
    n_timepoints: int = 12,
    return_y: bool = False,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Randomly generate 2D X and y data.

    Will ensure there is at least one sample per label if a classification
    label is being returned (return_y=True and regression_target=False).

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    n_timepoints : int
        The number of features/series length to generate.
    return_y : bool
        If True, return the labels as well as the data.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a scalar float, otherwise an int.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 2D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from aeon.datasets import make_example_2d_numpy
    >>> data, labels = make_example_2d_numpy(n_cases=2, n_timepoints=6,
    ...                                      return_y=True, n_labels=2, random_state=0)
    >>> print(data)
    [[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
     [2.         3.567092   3.85465104 1.53376608 3.16690015 2.11557968]]
    >>> print(labels)
    [0 1]
    """
    X, y = make_2d_test_data(
        n_cases=n_cases,
        n_timepoints=n_timepoints,
        n_labels=n_labels,
        regression_target=regression_target,
        random_state=random_state,
    )
    if return_y:
        return X, y
    return X


def make_example_long_table(n_cases=50, n_channels=2, n_timepoints=20):
    """Generate example from long table format file.

    Parameters
    ----------
    n_cases: int
        Number of cases.
    n_channels: int
        Number of dimensions.
    n_timepoints: int
        Length of the series.

    Returns
    -------
    DataFrame containing random data in long format.
    """
    rows_per_case = n_timepoints * n_channels
    total_rows = n_cases * n_timepoints * n_channels

    case_ids = np.empty(total_rows, dtype=int)
    idxs = np.empty(total_rows, dtype=int)
    dims = np.empty(total_rows, dtype=int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / n_timepoints)
        idxs[i] = rem % n_timepoints

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df


def make_example_multi_index_dataframe(n_instances=50, n_channels=3, n_timepoints=20):
    """Generate example multi-index DataFrame.

    Parameters
    ----------
    n_instances : int
        Number of instances.
    n_channels : int
        Number of columns (series) in multi-indexed DataFrame.
    n_timepoints : int
        Number of timepoints per instance-column pair.

    Returns
    -------
    mi_df : pd.DataFrame
        The multi-indexed DataFrame with
        shape (n_instances*n_timepoints, n_column).
    """
    # Make long DataFrame
    long_df = make_example_long_table(
        n_cases=n_instances, n_timepoints=n_timepoints, n_channels=n_channels
    )
    # Make Multi index DataFrame
    mi_df = long_df.set_index(["case_id", "reading_id"]).pivot(columns="dim_id")
    mi_df.columns = [f"var_{i}" for i in range(n_channels)]
    return mi_df


def _convert_tsf_to_hierarchical(
    data: pd.DataFrame,
    metadata: Dict,
    freq: str = None,
    value_column_name: str = "series_value",
) -> pd.DataFrame:
    """Convert the data from default_tsf to pd_multiindex_hier.

    Parameters
    ----------
    data : pd.DataFrame
        nested values dataframe
    metadata : Dict
        tsf file metadata
    freq : str, optional
        pandas compatible time frequency, by default None
        if not speciffied it's automatically mapped from the tsf frequency to a pandas
        frequency
    value_column_name: str, optional
        The name of the column that contains the values, by default "series_value"

    Returns
    -------
    pd.DataFrame
        aeon pd_multiindex_hier mtype
    """
    df = data.copy()

    if freq is None:
        freq_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
            "yearly": "YS",
        }
        freq = freq_map[metadata["frequency"]]

    # create the time index
    if "start_timestamp" in df.columns:
        df["timestamp"] = df.apply(
            lambda x: pd.date_range(
                start=x["start_timestamp"], periods=len(x[value_column_name]), freq=freq
            ),
            axis=1,
        )
        drop_columns = ["start_timestamp"]
    else:
        df["timestamp"] = df.apply(
            lambda x: pd.RangeIndex(start=0, stop=len(x[value_column_name])), axis=1
        )
        drop_columns = []

    # pandas implementation of multiple column explode
    # can be removed and replaced by explode if we move to pandas version 1.3.0
    columns = [value_column_name, "timestamp"]
    index_columns = [c for c in list(df.columns) if c not in drop_columns + columns]
    result = pd.DataFrame({c: df[c].explode() for c in columns})
    df = df.drop(columns=columns + drop_columns).join(result)
    if df["timestamp"].dtype == "object":
        df = df.astype({"timestamp": "int64"})
    df = df.set_index(index_columns + ["timestamp"])
    df = df.astype({value_column_name: "float"}, errors="ignore")

    return df
