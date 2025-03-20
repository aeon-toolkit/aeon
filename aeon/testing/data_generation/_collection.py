"""Utility functions for generating collections of time series."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "make_example_3d_numpy",
    "make_example_2d_numpy_collection",
    "make_example_3d_numpy_list",
    "make_example_2d_numpy_list",
    "make_example_dataframe_list",
    "make_example_2d_dataframe_collection",
    "make_example_multi_index_dataframe",
]

from typing import Union

import numpy as np
import pandas as pd


def make_example_3d_numpy(
    n_cases: int = 10,
    n_channels: int = 1,
    n_timepoints: int = 12,
    n_labels: int = 2,
    min_cases_per_label: int = 1,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Randomly generate 3D numpy X and numpy y data for testing.

    Generates data in 'numpy3D' format.

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
    min_cases_per_label : int
        The minimum number of samples per unique label.
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
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon.utils.validation.collection import get_type
    >>> data, labels = make_example_3d_numpy(
    ...     n_cases=2,
    ...     n_channels=2,
    ...     n_timepoints=6,
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
    >>> get_type(data)
    'numpy3D'
    """
    rng = np.random.RandomState(random_state)
    X = n_labels * rng.uniform(size=(n_cases, n_channels, n_timepoints))
    y = X[:, 0, 0].astype(int)

    for i in range(n_labels):
        for j in range(min_cases_per_label):
            idx = i * min_cases_per_label + j
            if len(y) > idx:
                X[idx, 0, 0] = i
                y[idx] = i
    X = X * (y[:, None, None] + 1)

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    if return_y:
        return X, y
    return X


def make_example_2d_numpy_collection(
    n_cases: int = 10,
    n_timepoints: int = 8,
    n_labels: int = 2,
    min_cases_per_label: int = 1,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Randomly generate 2D numpy X and numpy y for testing.

    Generates data in 'numpy2D' format.

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
    min_cases_per_label : int
        The minimum number of samples per unique label.
    regression_target : bool
        If True, the target will be a scalar float, otherwise an int.
    random_state : int or None
        Seed for random number generation.
    return_y : bool, default = True
        If True, return the labels as well as the data.

    Returns
    -------
    X : np.ndarray
        Randomly generated 2D data.
    y : np.ndarray
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_2d_numpy_collection
    >>> from aeon.utils.validation.collection import get_type
    >>> data, labels = make_example_2d_numpy_collection(
    ...     n_cases=2,
    ...     n_timepoints=6,
    ...     n_labels=2,
    ...     random_state=0,
    ... )
    >>> print(data)
    [[0.         1.43037873 1.20552675 1.08976637 0.8473096  1.29178823]
     [2.         3.567092   3.85465104 1.53376608 3.16690015 2.11557968]]
    >>> print(labels)
    [0 1]
    >>> get_type(data)
    'numpy2D'
    """
    rng = np.random.RandomState(random_state)
    X = n_labels * rng.uniform(size=(n_cases, n_timepoints))
    y = X[:, 0].astype(int)

    for i in range(n_labels):
        for j in range(min_cases_per_label):
            idx = i * min_cases_per_label + j
            if len(y) > idx:
                X[idx, 0] = i
                y[idx] = i
    X = X * (y[:, None] + 1)

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    if return_y:
        return X, y
    return X


def make_example_3d_numpy_list(
    n_cases: int = 10,
    n_channels: int = 1,
    min_n_timepoints: int = 8,
    max_n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[list[np.ndarray], tuple[list[np.ndarray], np.ndarray]]:
    """Randomly generate 3D list of numpy X and numpy y for testing.

    Generates data in 'np-list' format.

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
        Randomly generated potentially unequal length 3D data.
    y : np.ndarray
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy_list
    >>> from aeon.utils.validation.collection import get_type
    >>> data, labels = make_example_3d_numpy_list(
    ...     n_cases=2,
    ...     n_channels=2,
    ...     min_n_timepoints=4,
    ...     max_n_timepoints=6,
    ...     n_labels=2,
    ...     random_state=0,
    ... )
    >>> print(data)  # doctest: +NORMALIZE_WHITESPACE
    [array([[0.        , 1.6885315 , 1.71589124, 1.69450348],
           [1.24712739, 0.76876341, 0.59506921, 0.11342595]]),
           array([[2.        , 3.16690015, 2.11557968, 2.27217824],
           [3.70238655, 0.28414423, 0.3485172 , 0.08087359]])]
    >>> print(labels)
    [0 1]
    >>> get_type(data)
    'np-list'
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


def make_example_2d_numpy_list(
    n_cases: int = 10,
    min_n_timepoints: int = 8,
    max_n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[list[np.ndarray], tuple[list[np.ndarray], np.ndarray]]:
    """Randomly generate 2D list of numpy X and numpy y for testing.

    Will ensure there is at least one sample per label if a classification
    label is being returned (regression_target=False).

    Parameters
    ----------
    n_cases : int, default = 10
        The number of samples to generate.
    min_n_timepoints : int, default = 8
        The minimum number of features/series length to generate for individual series.
    max_n_timepoints : int, default = 12
        The maximum number of features/series length to generate for individual series.
    n_labels : int, default = 2
        The number of unique labels to generate.
    regression_target : bool, default = False
        If True, the target will be a scalar float, otherwise an int.
    random_state : int or None, default = None
        Seed for random number generation.
    return_y : bool, default = True
        Return the y target variable.

    Returns
    -------
    X : list of np.ndarray
        Randomly generated potentially unequal length 2D data.
    y : np.ndarray
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_2d_numpy_list
    >>> from aeon.utils.validation.collection import get_type
    >>> data, labels = make_example_2d_numpy_list(
    ...     n_cases=2,
    ...     min_n_timepoints=4,
    ...     max_n_timepoints=6,
    ...     n_labels=2,
    ...     random_state=0,
    ... )
    >>> print(data)  # doctest: +NORMALIZE_WHITESPACE
    [array([0.        , 1.6885315 , 1.71589124, 1.69450348]),
            array([2.        , 1.19013843, 0.22685191, 1.09062518, 1.91066047])]
    >>> print(labels)
    [0 1]
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


def make_example_dataframe_list(
    n_cases: int = 10,
    n_channels: int = 1,
    min_n_timepoints: int = 8,
    max_n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[list[pd.DataFrame], tuple[list[pd.DataFrame], np.ndarray]]:
    """Randomly generate list of DataFrame X and numpy y for testing.

    Generates data in 'df-list' format.

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
    X : list of pd.DataFrame
        Randomly generated potentially unequal length 3D data.
    y : np.ndarray
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_dataframe_list
    >>> from aeon.utils.validation.collection import get_type
    >>> data, labels = make_example_dataframe_list(
    ...     n_cases=2,
    ...     n_channels=2,
    ...     min_n_timepoints=4,
    ...     max_n_timepoints=6,
    ...     n_labels=2,
    ...     random_state=0,
    ... )
    >>> print(data)
    [          0         1         2         3
    0  0.000000  1.688531  1.715891  1.694503
    1  1.247127  0.768763  0.595069  0.113426,           0         1         2         3
    0  2.000000  3.166900  2.115580  2.272178
    1  3.702387  0.284144  0.348517  0.080874]
    >>> print(labels)
    [0 1]
    >>> get_type(data)
    'df-list'
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

        X.append(pd.DataFrame(x, index=range(n_channels), columns=range(n_timepoints)))
        y[i] = label

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    if return_y:
        return X, y
    return X


def make_example_2d_dataframe_collection(
    n_cases: int = 10,
    n_timepoints: int = 8,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
    return_y: bool = True,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, np.ndarray]]:
    """Randomly generate 2D DataFrame X and numpy y for testing.

    Generates data in 'pd-wide' format.

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
    X : pd.DataFrame
        Randomly generated 2D data.
    y : np.ndarray
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_2d_dataframe_collection
    >>> from aeon.utils.validation.collection import get_type
    >>> data, labels = make_example_2d_dataframe_collection(
    ...     n_cases=2,
    ...     n_timepoints=6,
    ...     n_labels=2,
    ...     random_state=0,
    ... )
    >>> print(data)
         0         1         2         3        4         5
    0  0.0  1.430379  1.205527  1.089766  0.84731  1.291788
    1  2.0  3.567092  3.854651  1.533766  3.16690  2.115580
    >>> print(labels)
    [0 1]
    >>> get_type(data)
    'pd-wide'
    """
    X, y = make_example_2d_numpy_collection(
        n_cases=n_cases,
        n_timepoints=n_timepoints,
        n_labels=n_labels,
        regression_target=regression_target,
        random_state=random_state,
        return_y=True,
    )
    X = pd.DataFrame(X, index=range(n_cases), columns=range(n_timepoints))

    if return_y:
        return X, y
    return X


def make_example_multi_index_dataframe(
    n_cases: int = 10,
    n_channels: int = 1,
    min_n_timepoints: int = 8,
    max_n_timepoints: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state=None,
    return_y: bool = True,
):
    """Randomly generate multi-index pd.DataFrame X and numpy y data for testing.

    Generates data in 'pd-multiindex' format.

    Will ensure there is at least one sample per label if a classification
    label is being returned (regression_target=False).

    Parameters
    ----------
    n_cases : int, default = 10
        The number of samples to generate.
    n_channels : int, default = 1
        The number of series channels to generate.
    min_n_timepoints : int, default = 12
        The minimum number of features/series length to generate for individual series.
    max_n_timepoints : int, default = 12
        The maximum number of features/series length to generate for individual series.
    n_labels : int, default = 2
        The number of unique labels to generate.
    regression_target : bool, default = False
        If True, the target will be a float, otherwise a discrete.
    random_state : int or None, default = None
        Seed for random number generation.
    return_y : bool, default = True
        Return the y target variable.

    Returns
    -------
    X : pd.DataFrame
        Randomly generated potentially unequal length 3D data.
    y : np.ndarray
        Randomly generated labels if return_y is True.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_multi_index_dataframe
    >>> from aeon.utils.validation.collection import get_type
    >>> data, labels = make_example_multi_index_dataframe(
    ...     n_cases=2,
    ...     n_channels=2,
    ...     min_n_timepoints=4,
    ...     max_n_timepoints=6,
    ...     n_labels=2,
    ...     random_state=0,
    ... )
    >>> print(data)  # doctest: +NORMALIZE_WHITESPACE
    channel                0         1
    case timepoint
    0    0          0.000000  1.247127
         1          1.688531  0.768763
         2          1.715891  0.595069
         3          1.694503  0.113426
    1    0          2.000000  3.702387
         1          3.166900  0.284144
         2          2.115580  0.348517
         3          2.272178  0.080874
    >>> print(labels)
    [0 1]
    >>> get_type(data)
    'pd-multiindex'
    """
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame()
    X["case"] = pd.Series(dtype=np.int32)
    X["channel"] = pd.Series(dtype=np.int32)
    X["timepoint"] = pd.Series(dtype=np.int32)
    X["value"] = pd.Series(dtype=np.float32)
    y = np.zeros(n_cases, dtype=np.int32)

    for i in range(n_cases):
        n_timepoints = rng.randint(min_n_timepoints, max_n_timepoints + 1)
        x = n_labels * rng.uniform(size=(n_channels, n_timepoints))
        label = x[0, 0].astype(int)
        if i < n_labels and n_cases > i:
            x[0, 0] = i
            label = i
        x = x * (label + 1)

        df = pd.DataFrame()
        df["case"] = pd.Series([i] * n_channels * n_timepoints)
        df["channel"] = pd.Series(np.repeat(range(n_channels), n_timepoints))
        df["timepoint"] = pd.Series(np.tile(range(n_timepoints), n_channels))
        df["value"] = pd.Series(x.reshape(-1))

        X = pd.concat([X, df])
        y[i] = label

    X = X.reset_index(drop=True)
    X = X.pivot(index=["case", "timepoint"], columns=["channel"], values="value")

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    if return_y:
        return X, y
    return X
