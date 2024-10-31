"""Data generators."""

from typing import Optional

__maintainer__ = []


import pandas as pd


def _convert_tsf_to_hierarchical(
    data: pd.DataFrame,
    metadata: dict,
    freq: Optional[str] = None,
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
        hierarchical multiindex pd.Dataframe
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
