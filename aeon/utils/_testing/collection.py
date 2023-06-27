# -*- coding: utf-8 -*-
"""Utility functions for generating collections of time series."""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "make_3d_test_data",
    "make_2d_test_data",
    "make_unequal_length_test_data",
    "make_nested_dataframe_data",
    "make_clustering_data",
]

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from aeon.datatypes import convert


def make_3d_test_data(
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
    >>> from aeon.utils._testing.collection import make_3d_test_data
    >>> data, labels = make_3d_test_data(
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


def make_2d_test_data(
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
    >>> from aeon.utils._testing.collection import make_2d_test_data
    >>> data, labels = make_2d_test_data(
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


def make_unequal_length_test_data(
    n_cases: int = 10,
    n_channels: int = 1,
    min_series_length: int = 6,
    max_series_length: int = 8,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
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
    min_series_length : int
        The minimum number of features/series length to generate for invidiaul series.
    max_series_length : int
        The maximum number of features/series length to generate for invidiaul series.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a scalar float, otherwise an int.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : list of np.ndarray
        Randomly generated unequal length 3D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from aeon.utils._testing.collection import make_unequal_length_test_data
    >>> data, labels = make_unequal_length_test_data(
    ...     n_cases=20,
    ...     n_channels=2,
    ...     min_series_length=8,
    ...     max_series_length=12,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = []
    y = np.zeros(n_cases, dtype=np.int32)

    for i in range(n_cases):
        series_length = rng.randint(min_series_length, max_series_length + 1)
        x = n_labels * rng.uniform(size=(n_channels, series_length))
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

    return X, y


def _make_collection(
    n_instances=20,
    n_channels=1,
    n_timepoints=20,
    y=None,
    all_positive=False,
    random_state=None,
    return_type="numpy3D",
):
    """Generate aeon compatible test data, data formats.

    Parameters
    ----------
    n_instances : int, optional, default=20
        number of instances per series in the collection
    n_channels : int, optional, default=1
        number of variables in the time series
    n_timepoints : int, optional, default=20
        number of time points in each series
    y : None (default), or 1D np.darray or 1D array-like, shape (n_cases, )
        if passed, return will be generated with association to y
    all_positive : bool, optional, default=False
        whether series contain only positive values when generated
    random_state : None (default) or int
        if int is passed, will be used in numpy RandomState for generation
    return_type : str, aeon collection type, default="numpy3D"

    Returns
    -------
    X : an aeon time series data container of type return_type
        with n_cases instances, n_channels variables, n_timepoints time points
        generating distribution is all values i.i.d. normal with std 0.5
        if y is passed, i-th series values are additively shifted by y[i] * 100
    """
    # If target variable y is given, we ignore n_cases and instead generate as
    # many instances as in the target variable
    if y is not None:
        y = np.asarray(y)
        n_instances = len(y)
    rng = check_random_state(random_state)

    # Generate data as 3d numpy array
    X = rng.normal(scale=0.5, size=(n_instances, n_channels, n_timepoints))

    # Generate association between data and target variable
    if y is not None:
        X = X + (y * 100).reshape(-1, 1, 1)

    if all_positive:
        X = X**2

    X = convert(X, from_type="numpy3D", to_type=return_type)
    return X


def _make_collection_X(
    n_instances=20,
    n_channels=1,
    n_timepoints=20,
    y=None,
    all_positive=False,
    return_numpy=False,
    random_state=None,
):
    if return_numpy:
        return_type = "numpy3D"
    else:
        return_type = "nested_univ"

    return _make_collection(
        n_instances=n_instances,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        y=y,
        all_positive=all_positive,
        random_state=random_state,
        return_type=return_type,
    )


def _make_regression_y(n_instances=20, return_numpy=True, random_state=None):
    rng = check_random_state(random_state)
    y = rng.normal(size=n_instances)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def _make_classification_y(
    n_instances=20, n_classes=2, return_numpy=True, random_state=None
):
    if not n_instances >= n_classes:
        raise ValueError("n_cases must be bigger than n_classes")
    rng = check_random_state(random_state)
    n_repeats = int(np.ceil(n_instances / n_classes))
    y = np.tile(np.arange(n_classes), n_repeats)[:n_instances]
    rng.shuffle(y)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def make_nested_dataframe_data(
    n_cases: int = 20,
    n_channels: int = 1,
    n_timepoints: int = 20,
    n_classes: int = 2,
    classification: bool = True,
    random_state=None,
):
    """Randomly generate nest pd.DataFrame X and pd.Series y data for testing.

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    n_channels : int
        The number of series channels to generate.
    n_timepoints : int
        The number of features/series length to generate.
    classification : bool
        If True, the target will be discrete, otherwise a float.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 3D data.
    y : np.ndarray
        Randomly generated labels.
    """
    if classification:
        """Make Classification Problem."""
        y = _make_classification_y(
            n_cases, n_classes, return_numpy=False, random_state=random_state
        )
    else:
        y = _make_regression_y(n_cases, return_numpy=False, random_state=random_state)
    X = _make_collection_X(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        return_numpy=False,
        random_state=random_state,
        y=y,
    )

    return X, y


def make_clustering_data(
    n_cases: int = 20,
    n_channels: int = 1,
    n_timepoints: int = 20,
    return_numpy: bool = True,
    random_state=None,
):
    """Randomly generate nest pd.DataFrame X and pd.Series y data for testing.

    Parameters
    ----------
    n_cases : int
        The number of samples to generate.
    n_channels : int
        The number of series channels to generate.
    n_timepoints : int
        The number of features/series length to generate.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 3D data.
    y : np.ndarray
        Randomly generated labels.
    """
    X, _ = make_3d_test_data(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=random_state,
    )
    return X


def _make_nested_from_array(array, n_instances=20, n_columns=1):
    return pd.DataFrame(
        [[pd.Series(array) for _ in range(n_columns)] for _ in range(n_instances)],
        columns=[f"col{c}" for c in range(n_columns)],
    )
