# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
import pandas as pd


def make_example_long_table(n_cases=50, n_channels=2, series_length=20):
    """Generate example from long table format file.

    Parameters
    ----------
    n_cases: int
        Number of cases.
    series_length: int
        Length of the series.
    n_channels: int
        Number of dimensions.

    Returns
    -------
    DataFrame
    """
    rows_per_case = series_length * n_channels
    total_rows = n_cases * series_length * n_channels

    case_ids = np.empty(total_rows, dtype=int)
    idxs = np.empty(total_rows, dtype=int)
    dims = np.empty(total_rows, dtype=int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / series_length)
        idxs[i] = rem % series_length

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df


def make_example_multi_index_dataframe(n_instances=50, n_channels=3, series_length=20):
    """Generate example multi-index DataFrame.

    Parameters
    ----------
    n_instances : int
        Number of instances.
    n_channels : int
        Number of columns (series) in multi-indexed DataFrame.
    series_length : int
        Number of timepoints per instance-column pair.

    Returns
    -------
    mi_df : pd.DataFrame
        The multi-indexed DataFrame with
        shape (n_instances*n_timepoints, n_column).
    """
    # Make long DataFrame
    long_df = make_example_long_table(
        n_cases=n_instances, series_length=series_length, n_channels=n_channels
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
