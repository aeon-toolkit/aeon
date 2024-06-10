import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from aeon.utils.conversion import convert_collection


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
