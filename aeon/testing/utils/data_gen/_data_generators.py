"""Data generators."""

__maintainer__ = []

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


# TODO: Move to the conversion module after #1125
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


def _make_3d_test_data(
    n_cases: int = 10,
    n_channels: int = 1,
    n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate 3D X and y data for testing.

    Will ensure there is at least one sample per label if a classification
    label is being returned (regression_target=False).

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    n_channels : int
        The number of series channels to generate.
    n_timepoints : int
        The number of features/series length to generate.
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
    >>> from aeon.testing.utils.data_gen import make_example_3d_numpy
    >>> data, labels = make_example_3d_numpy(
    ...     n_cases=20,
    ...     n_channels=2,
    ...     n_timepoints=10,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = n_labels * rng.uniform(size=(n_cases, n_channels, n_timepoints))
    y = X[:, 0, 0].astype(int)

    for i in range(n_labels):
        if len(y) > i:
            X[i, 0, 0] = i
            y[i] = i
    X = X * (y[:, None, None] + 1)

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    return X, y


def _make_2d_test_data(
    n_cases: int = 10,
    n_timepoints: int = 8,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate 2D data for testing.

    Will ensure there is at least one sample per label if a classification
    label is being returned (regression_target=False).

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    n_timepoints : int
        The number of features/series length to generate.
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
    >>> from aeon.testing.utils.data_gen import make_example_3d_numpy
    >>> data, labels = make_example_3d_numpy(
    ...     n_cases=20,
    ...     n_timepoints=10,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = n_labels * rng.uniform(size=(n_cases, n_timepoints))
    y = X[:, 0].astype(int)

    for i in range(n_labels):
        if len(y) > i:
            X[i, 0] = i
            y[i] = i
    X = X * (y[:, None] + 1)

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    return X, y
