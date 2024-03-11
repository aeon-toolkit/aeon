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


def make_example_2d_unequal_length(
    n_cases: int = 10,
    min_n_timepoints: int = 6,
    max_n_timepoints: int = 8,
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
        The minimum number of features/series length to generate for invidiaul series.
    max_n_timepoints : int
        The maximum number of features/series length to generate for invidiaul series.
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


def make_example_unequal_length(
    n_cases: int = 10,
    n_channels: int = 1,
    min_n_timepoints: int = 6,
    max_n_timepoints: int = 8,
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
        The minimum number of features/series length to generate for invidiaul series.
    max_n_timepoints : int
        The maximum number of features/series length to generate for invidiaul series.
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
    >>> from aeon.testing.utils.data_gen import make_example_unequal_length
    >>> data, labels = make_example_unequal_length(
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


def make_example_nested_dataframe(
    n_cases: int = 20,
    n_channels: int = 1,
    n_timepoints: int = 20,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state=None,
    return_y: bool = True,
):
    """Randomly generate nest pd.DataFrame X and pd.Series y data for testing.

    Parameters
    ----------
    n_cases : int, default = 20
        The number of samples to generate.
    n_channels : int, default = 1
        The number of series channels to generate.
    n_timepoints : int, default = 20
        The number of features/series length to generate.
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
    X : np.ndarray
        Randomly generated 3D data.
    y : np.ndarray
        Randomly generated labels.
    """
    X = _make_collection_X(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        return_numpy=False,
        random_state=random_state,
    )
    if return_y:
        if not regression_target:
            """Make Classification Problem."""
            y = _make_classification_y(
                n_cases, n_labels, return_numpy=False, random_state=random_state
            )
        else:
            y = _make_regression_y(
                n_cases, return_numpy=False, random_state=random_state
            )
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


def _make_collection(
    n_cases=20,
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
    n_cases : int, optional, default=20
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
        n_cases = len(y)
    rng = check_random_state(random_state)

    # Generate data as 3d numpy array
    X = rng.normal(scale=0.5, size=(n_cases, n_channels, n_timepoints))

    # Generate association between data and target variable
    if y is not None:
        X = X + (y * 100).reshape(-1, 1, 1)

    if all_positive:
        X = X**2

    X = convert_collection(X, return_type)
    return X


def _make_collection_X(
    n_cases=20,
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
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        y=y,
        all_positive=all_positive,
        random_state=random_state,
        return_type=return_type,
    )


def _make_regression_y(n_cases=20, return_numpy=True, random_state=None):
    rng = check_random_state(random_state)
    y = rng.normal(size=n_cases)
    y = y.astype(np.float32)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def _make_classification_y(
    n_cases=20, n_classes=2, return_numpy=True, random_state=None
):
    if not n_cases >= n_classes:
        raise ValueError("n_cases must be bigger than n_classes")
    rng = check_random_state(random_state)
    n_repeats = int(np.ceil(n_cases / n_classes))
    y = np.tile(np.arange(n_classes), n_repeats)[:n_cases]
    rng.shuffle(y)
    if return_numpy:
        return y
    else:
        return pd.Series(y)


def _make_nested_from_array(array, n_cases=20, n_columns=1):
    return pd.DataFrame(
        [[pd.Series(array) for _ in range(n_columns)] for _ in range(n_cases)],
        columns=[f"col{c}" for c in range(n_columns)],
    )


np_list = []
for _ in range(10):
    np_list.append(np.random.random(size=(1, 20)))
df_list = []
for _ in range(10):
    df_list.append(pd.DataFrame(np.random.random(size=(20, 1))))
nested, _ = make_example_nested_dataframe(n_cases=10)
multiindex = make_example_multi_index_dataframe(
    n_cases=10, n_channels=1, n_timepoints=20
)

EQUAL_LENGTH_UNIVARIATE = {
    "numpy3D": np.random.random(size=(10, 1, 20)),
    "np-list": np_list,
    "df-list": df_list,
    "numpy2D": np.zeros(shape=(10, 20)),
    "pd-wide": pd.DataFrame(np.zeros(shape=(10, 20))),
    "nested_univ": nested,
    "pd-multiindex": multiindex,
}
np_list_uneq = []
for i in range(10):
    np_list_uneq.append(np.random.random(size=(1, 20 + i)))
df_list_uneq = []
for i in range(10):
    df_list_uneq.append(pd.DataFrame(np.random.random(size=(20 + i, 1))))

nested_univ_uneq = pd.DataFrame(dtype=float)
instance_list = []
for i in range(0, 10):
    instance_list.append(pd.Series(np.random.randn(20 + i)))
nested_univ_uneq["channel0"] = instance_list

UNEQUAL_LENGTH_UNIVARIATE = {
    "np-list": np_list_uneq,
    "df-list": df_list_uneq,
    "nested_univ": nested_univ_uneq,
}
np_list_multi = []
for _ in range(10):
    np_list_multi.append(np.random.random(size=(2, 20)))
df_list_multi = []
for _ in range(10):
    df_list_multi.append(pd.DataFrame(np.random.random(size=(20, 2))))
multi = make_example_multi_index_dataframe(n_cases=10, n_channels=2, n_timepoints=20)

nested_univ_multi = pd.DataFrame(dtype=float)
instance_list = []
for _ in range(0, 10):
    instance_list.append(pd.Series(np.random.randn(20)))
nested_univ_multi["channel0"] = instance_list
instance_list = []
for _ in range(0, 10):
    instance_list.append(pd.Series(np.random.randn(20)))
nested_univ_multi["channel1"] = instance_list


EQUAL_LENGTH_MULTIVARIATE = {
    "numpy3D": np.random.random(size=(10, 2, 20)),
    "np-list": np_list_multi,
    "df-list": df_list_multi,
    "nested_univ": nested_univ_multi,
    "pd-multiindex": multi,
}
