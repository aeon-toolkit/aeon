"""Functions to write aeon datasets to file."""

__maintainers__ = ["MatthewMiddlehurst"]
__all__ = ["save_to_ts_file"]

import os
import textwrap

import numpy as np

from aeon.utils.conversion import convert_collection
from aeon.utils.validation.collection import (
    get_n_cases,
    get_n_channels,
    get_n_timepoints,
    has_missing,
    is_collection,
    is_equal_length,
)
from aeon.utils.validation.labels import check_classification_y, check_regression_y


def save_to_ts_file(
    X,
    y=None,
    *,
    label_type=None,
    path="./",
    problem_name="data",
    file_suffix=None,
    header=None,
):
    """Write an ``aeon`` collection of time series to text file in ``.ts`` format.

    Write metadata and data stored in aeon compatible data set to file.
    A description of the ``.ts`` format available at
    https://www.aeon-toolkit.org/en/stable/api_reference/data_format.html.

    Parameters
    ----------
    X: ``aeon`` collection data format
        A collection of time series cases in one of the following formats:

        - "numpy3D": a 3D numpy ndarray of shape ``(n_cases, n_channels, n_timepoints)``
        - "np-list":  length ``n_cases`` Python list of 2D numpy ndarray
            with shape ``(n_channels, n_timepoints_i)``
        - "df-list":  length ``n_cases`` Python list of 2D pandas DataFrame
            with shape (n_channels, n_timepoints_i)
        - "numpy2D":  a 2D numpy ndarray of shape ``(n_cases, n_timepoints)``
        - "pd-wide":  a 2D pandas DataFrame of shape ``(n_cases, n_timepoints)``
        - "pd-multiindex": a pandas DataFrame with MultiIndex, index
            ``[case, timepoint]``, columns ``[channel]``
    y:  np.ndarray, pd.Series or None, default=None
        The response variable if present. Must be discrete for classification,
        continuous for regression. ``None`` if no labels are written.
    label_type: str or None, default=None
        If not ``None``, specifies the type of label, either ``"classification"`` or
        ``"regression"`` to ensure the correct header is written. Must be set if
        ``y`` is not ``None``.
    path: str, default="./"
        The directory to write the file to. If the directory does not exist, it will be
        created.
    problem_name: string, default = "data"
        The name of the problem being written to file. Used in the file metadata and
        file name.
        The file is written to ``{path}/{problem_name}{file_suffix}.ts``.
    file_suffix: str or None, default=None
        If not ``None``, this string is appended to the end of the file name, i.e.
        ``"_TRAIN"`` or ``"_TEST"``.
        The file is written to ``{path}/{problem_name}{file_suffix}.ts``.
    header: str or None, default=None
        Optional text at the top of the written file. This is for information only and
        is ignored when loading.
    """
    if not is_collection(X, include_2d=True):
        raise TypeError(
            "Wrong input data type for X. Convert to an aeon collection format, "
            "e.g. numpy3D (n_cases, n_channels, n_timepoints) or np-list of "
            "length n_cases containing np.ndarray's of shape "
            "(n_channels, n_timepoints_i) if unequal length."
        )

    X = convert_collection(X, "np-list")

    n_cases = get_n_cases(X)
    n_channels = get_n_channels(X)
    n_timepoints = get_n_timepoints(X)
    univariate = n_channels == 1
    equal_length = is_equal_length(X)
    has_missing_values = has_missing(X)

    bad_label_type = (
        "If y is not None, label_type must be either 'classification' or 'regression'."
    )
    if y is None:
        target_metadata = "@targetlabel false"
    elif isinstance(label_type, str):
        label_type = label_type.lower()
        if label_type == "classification":
            check_classification_y(y)
            class_labels = np.unique(y)
            space_separated_class_label = " ".join(str(label) for label in class_labels)
            target_metadata = f"@classLabel true {space_separated_class_label}"
        elif label_type == "regression":
            check_regression_y(y)
            target_metadata = "@targetlabel true"
        else:
            raise ValueError(bad_label_type)

        if n_cases != len(y):
            raise ValueError(
                "The number of cases in X does not match the number of values in y."
            )
    else:
        raise ValueError(bad_label_type)

    # create dir
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        raise ValueError(f"Error trying to create {path}.")

    file_suffix = "" if file_suffix is None else file_suffix
    with open(os.path.join(path, f"{problem_name}{file_suffix}.ts"), "w") as file:
        # write header and metadata
        if header is not None:
            file.write("\n# ".join(textwrap.wrap("# " + header)))
            file.write("\n")

        file.write(f"@problemName {problem_name}\n")
        file.write("@timestamps false\n")
        file.write(f"@missing {has_missing_values}\n")
        file.write(f"@univariate {str(univariate).lower()}\n")
        if not univariate:
            file.write(f"@dimension {n_channels}\n")
        file.write(f"@equalLength {str(equal_length).lower()}\n")
        if equal_length:
            file.write(f"@seriesLength {n_timepoints}\n")
        file.write(f"{target_metadata}\n")

        # start writing data
        file.write("@data\n")
        for i in range(n_cases):
            for j in range(n_channels):
                series = ",".join(
                    [str(num) if not np.isnan(num) else "NaN" for num in X[i][j]]
                )
                file.write(str(series))
                file.write(":")
            if y is not None:
                file.write(str(y[i]))
            file.write("\n")
        file.close()
