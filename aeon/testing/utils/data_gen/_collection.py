"""Utility functions for generating collections of time series."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from aeon.utils.conversion import convert_collection


def make_example_3d_numpy(
    n_cases: int = 10,
    n_channels: int = 1,
    n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    return_y : bool, default = True
        Return the y target variable.

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
    ...     n_cases=2,
    ...     n_channels=2,
    ...     n_timepoints=6,
    ...     return_y=True,
    ...     n_labels=2,
    ...     random_state=0,
    ... )
    >>> print(data)
    [[[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
      [0.87517442 1.783546   1.92732552 0.76688304 1.58345008 1.05778984]]
    <BLANKLINE>
     [[2.         3.70238655 0.28414423 0.3485172  0.08087359 3.33047938]
      [3.112627   3.48004859 3.91447337 3.19663426 1.84591745 3.12211671]]]
    >>> print(labels)
    [0 1]
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
    if return_y:
        return X, y
    return X


def make_example_2d_numpy(
    n_cases: int = 10,
    n_timepoints: int = 8,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    return_y : bool, default = True
        If True, return the labels as well as the data.

    Returns
    -------
    X : np.ndarray
        Randomly generated 1D data.
    y : np.ndarray
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> data, labels = make_example_2d_numpy(n_cases=2, n_timepoints=6,
    ... n_labels=2, random_state=0)
    >>> print(data)
    [[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
     [2.         3.567092   3.85465104 1.53376608 3.16690015 2.11557968]]
    >>> print(labels)
    [0 1]
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
    if return_y:
        return X, y
    return X


def make_example_3d_unequal_length(
    n_cases: int = 10,
    n_channels: int = 1,
    min_n_timepoints: int = 8,
    max_n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Randomly generate unequal length X and y for testing.

    Will ensure there is at least one sample per label if a classification
    label is being returned (regression_target=False).

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    n_channels : int
        The number of series channels to generate.
    min_n_timepoints : int
        The minimum number of features/series length to generate for individual series.
    max_n_timepoints : int
        The maximum number of features/series length to generate for individual series.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a scalar float, otherwise an int.
    random_state : int or None
        Seed for random number generation.
    return_y : bool, default = True
        Return the y target variable.

    Returns
    -------
    X : list of np.ndarray
        Randomly generated unequal length 3D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from aeon.testing.utils.data_gen import make_example_3d_unequal_length
    >>> data, labels = make_example_3d_unequal_length(
    ...     n_cases=20,
    ...     n_channels=2,
    ...     min_n_timepoints=8,
    ...     max_n_timepoints=12,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = []
    y = np.zeros(n_cases, dtype=np.int32)

    for i in range(n_cases):
        n_timepoints = rng.randint(min_n_timepoints, max_n_timepoints + 1)
        x = n_labels * rng.uniform(size=(n_channels, n_timepoints))
        label = x[0, 0].astype(int)
        if i < n_labels and n_cases > i:
            x[0, 0] = i
            label = i
        x = x * (label + 1)

        X.append(x)
        y[i] = label

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    if return_y:
        return X, y
    return X


def make_example_2d_unequal_length(
    n_cases: int = 10,
    min_n_timepoints: int = 8,
    max_n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """Randomly generate 2D unequal length X and y for testing.

    Will ensure there is at least one sample per label if a classification
    label is being returned (regression_target=False).

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    min_n_timepoints : int
        The minimum number of features/series length to generate for individual series.
    max_n_timepoints : int
        The maximum number of features/series length to generate for individual series.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a scalar float, otherwise an int.
    random_state : int or None
        Seed for random number generation.
    return_y : bool, default = True
        Return the y target variable.

    Returns
    -------
    X : list of np.ndarray
        Randomly generated unequal length 2D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from aeon.testing.utils.data_gen import make_example_2d_unequal_length
    >>> data, labels = make_example_2d_unequal_length(
    ...     n_cases=20,
    ...     min_n_timepoints=8,
    ...     max_n_timepoints=12,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = []
    y = np.zeros(n_cases, dtype=np.int32)
    for i in range(n_cases):
        n_timepoints = rng.randint(min_n_timepoints, max_n_timepoints + 1)
        x = n_labels * rng.uniform(size=n_timepoints)
        label = x[0].astype(int)
        if i < n_labels and n_cases > i:
            x[0] = i
            label = i
        x = x * (label + 1)

        X.append(x)
        y[i] = label

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    if return_y:
        return X, y
    return X


def make_example_nested_dataframe(
    n_cases: int = 10,
    n_channels: int = 1,
    n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    unequal_length: bool = False,
    min_n_timepoints: int = 8,
    random_state=None,
    return_y: bool = True,
):
    """Randomly generate nest pd.DataFrame X and pd.Series y data for testing.

    Parameters
    ----------
    n_cases : int, default = 10
        The number of samples to generate.
    n_channels : int, default = 1
        The number of series channels to generate.
    n_timepoints : int, default = 12
        The number of features/series length to generate.
    n_labels : int, default = 2
        The number of unique labels to generate.
    regression_target : bool, default = False
        If True, the target will be a float, otherwise a discrete.
    unequal_length : bool, default = False
        If True, generate unequal length series.
    min_n_timepoints : int, default = 8
        The minimum number of features/series length to generate for individual series.
        Only used if unequal_length is True.
    random_state : int or None, default = None
        Seed for random number generation.
    return_y : bool, default = True
        Return the y target variable.

    Returns
    -------
    X : np.ndarray
        Randomly generated 3D data.
    y : np.ndarray
        Randomly generated labels.
    """
    X, y = make_example_3d_unequal_length(
        n_cases=n_cases,
        n_channels=n_channels,
        min_n_timepoints=min_n_timepoints if unequal_length else n_timepoints,
        max_n_timepoints=n_timepoints,
        n_labels=n_labels,
        regression_target=regression_target,
        random_state=random_state,
        return_y=True,
    )
    X = convert_collection(X, "nested_univ")

    if return_y:
        return X, y
    return X


def make_example_long_table(
    n_cases: int = 50, n_channels: int = 2, n_timepoints: int = 20
) -> pd.DataFrame:
    """Generate example collection in long table format file.

    Parameters
    ----------
    n_cases: int, default = 50
        Number of cases.
    n_channels: int, default = 2
        Number of dimensions.
    n_timepoints: int, default = 20
        Length of the series.

    Returns
    -------
    pd.DataFrame
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


def make_example_multi_index_dataframe(
    n_cases: int = 50, n_channels: int = 3, n_timepoints: int = 20
):
    """Generate example collection as multi-index DataFrame.

    Parameters
    ----------
    n_cases : int, default =50
        Number of instances.
    n_channels : int, default =3
        Number of columns (series) in multi-indexed DataFrame.
    n_timepoints : int, default =20
        Number of timepoints per instance-column pair.

    Returns
    -------
    mi_df : pd.DataFrame
        The multi-indexed DataFrame with
        shape (n_cases*n_timepoints, n_column).
    """
    # Make long DataFrame
    long_df = make_example_long_table(
        n_cases=n_cases, n_timepoints=n_timepoints, n_channels=n_channels
    )
    # Make Multi index DataFrame
    mi_df = long_df.set_index(["case_id", "reading_id"]).pivot(columns="dim_id")
    mi_df.columns = [f"var_{i}" for i in range(n_channels)]
    return mi_df
