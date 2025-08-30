"""Functions to write aeon datasets to file."""

__maintainers__ = ["MatthewMiddlehurst"]
__all__ = ["save_to_ts_file"]

import os
import textwrap

import numpy as np
from deprecated.sphinx import deprecated

from aeon.utils.conversion import convert_collection
from aeon.utils.validation import has_missing, is_equal_length
from aeon.utils.validation.collection import (
    get_n_cases,
    get_n_channels,
    get_n_timepoints,
    is_collection,
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


# TODO: remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="write_to_ts_file will be removed in v1.4.0. Use save_to_ts_file instead.",
    category=FutureWarning,
)
def write_to_ts_file(
    X,
    path,
    y=None,
    problem_name="sample_data.ts",
    header=None,
    regression=False,
):
    """Write an aeon collection of time series to text file in .ts format.

    Write metadata and data stored in aeon compatible data set to file.
    A description of the ts format is in examples/load_data.ipynb.

    Parameters
    ----------
    X : Union[list, np.ndarray]
        Collection of time series, either equal length shape
        `(n_cases, n_channels, n_timepoints)` or unequal length as a list of
        np.ndarray, each of shape `(n_channels,n_timepoints_i)`.
    path : string
        Location of the directory to write file
    y: None or np.ndarray, default = None
        Response variable, discrete for classification, continuous for regression
        None if clustering.
    problem_name : string, default = "sample_data"
        The file is written to <path>/<problem_name>/<problem_name>.ts
    header: string, default = None
        Optional text at the top of the file that is ignored when loading.
    regression: boolean, default = False
        Indicate if this is a regression problem, so it is correctly specified in
        the header since there is no definite way of inferring this from y
    """
    if not (isinstance(X, np.ndarray) or isinstance(X, list)):
        raise TypeError(
            f" Wrong input data type {type(X)} convert to np.ndarray ("
            f"n_cases, n_channels,n_timepoints) if equal length or list "
            f"of [n_cases] np.ndarray shape (n_channels, n_timepoints) if unequal"
        )

    # See if passed file name contains .ts extension or not
    problem_name, extension = os.path.splitext(problem_name)
    if extension == "":  # .ts file extension not present
        extension = ".ts"
    if extension is not None:
        problem_name = problem_name + extension

    class_labels = None
    if y is not None:
        # ensure number of cases is same as the class value list
        if len(X) != len(y):
            raise IndexError(
                "The number of cases in X does not match the number of values in y"
            )
    if not regression:
        class_labels = np.unique(y)
    n_cases = len(X)
    n_channels = len(X[0])
    univariate = n_channels == 1
    equal_length = True
    if isinstance(X, list):
        length = len(X[0][0])
        for i in range(1, n_cases):
            if length != len(X[i][0]):
                equal_length = False
                break
    n_timepoints = -1
    if equal_length:
        n_timepoints = len(X[0][0])
    file = _write_header(
        path,
        problem_name,
        univariate=univariate,
        equal_length=equal_length,
        n_timepoints=n_timepoints,
        class_labels=class_labels,
        comment=header,
        regression=regression,
        extension=None,
    )
    missing_values = "NaN"
    for i in range(n_cases):
        for j in range(n_channels):
            series = ",".join(
                [str(num) if not np.isnan(num) else missing_values for num in X[i][j]]
            )
            file.write(str(series))
            file.write(":")
        if y is not None:
            file.write(str(y[i]))
        file.write("\n")
    file.close()


def _write_header(
    path,
    problem_name,
    univariate=True,
    equal_length=False,
    n_timepoints=-1,
    comment=None,
    regression=False,
    class_labels=None,
    extension=None,
):
    if class_labels is not None and regression:
        raise ValueError("Cannot have class_labels true for a regression problem")
    # create path if it does not exist
    dir = os.path.join(path, "")
    try:
        os.makedirs(dir, exist_ok=True)
    except OSError:
        raise ValueError(f"Error trying to access {dir} in _write_header")
    # create ts file in the path
    load_path = os.path.join(dir, problem_name)
    file = open(load_path, "w")
    # write comment if any as a block at start of file
    if comment is not None:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")

    """ Writes the header info for a ts file"""
    file.write(f"@problemName {problem_name}\n")
    file.write("@timestamps false\n")
    file.write(f"@univariate {str(univariate).lower()}\n")
    file.write(f"@equalLength {str(equal_length).lower()}\n")
    if n_timepoints > 0 and equal_length:
        file.write(f"@seriesLength {n_timepoints}\n")
    # write class labels line
    if class_labels is not None:
        space_separated_class_label = " ".join(str(label) for label in class_labels)
        file.write(f"@classLabel true {space_separated_class_label}\n")
    else:
        file.write("@classLabel false\n")
        if regression:  # or if a regression problem, write target label
            file.write("@targetlabel true\n")
    file.write("@data\n")
    return file


# TODO: remove in v1.4.0
@deprecated(
    version="1.3.0",
    reason="write_to_arff_file will be removed in v1.4.0.",
    category=FutureWarning,
)
def write_to_arff_file(
    X,
    y,
    path,
    problem_name="sample_data",
    header=None,
):
    """Write an aeon collection of time series to text file in .arff format.

    Only compatible for classification-like problems with univariate equal
    length time series currently.

    Parameters
    ----------
    X : np.ndarray (n_cases, n_channels, n_timepoints)
        Collection of univariate time series with equal length.
    y: ndarray
        Discrete response variable.
    path : string.
        Location of the directory to write file
    problem_name: str, default="Data"
        The problem name to print in the header of the arff file and also the name of
        the file.
    header: string, default=None
        Optional text at the top of the file that is ignored when loading.

    Returns
    -------
    None
    """
    if not (isinstance(X, np.ndarray)):
        raise TypeError(
            f" Wrong input data type {type(X)}. Convert to np.ndarray (n_cases, "
            f"n_channels, n_timepoints) if possible."
        )

    if len(X.shape) != 3 or X.shape[1] != 1:
        raise ValueError(
            f"X must be a 3D array with shape (n_cases, 1, n_timepoints), but "
            f"received {X.shape}"
        )

    file = open(f"{path}/{problem_name}.arff", "w")

    # write comment if any as a block at start of file
    if header is not None:
        file.write("\n% ".join(textwrap.wrap("% " + header)))
        file.write("\n")

    # begin writing header information
    file.write(f"@Relation {problem_name}\n")

    # write each attribute
    for i in range(X.shape[2]):
        file.write(f"@attribute att{str(i)} numeric\n")

    # lass attribute if it exists
    comma_separated_class_label = ",".join(str(label) for label in np.unique(y))
    file.write(f"@attribute target {{{comma_separated_class_label}}}\n")

    # write data
    file.write("@data\n")
    for case, target in zip(X, y):
        # turn attributes into comma-separated row
        atts = ",".join([str(num) if not np.isnan(num) else "?" for num in case[0]])
        file.write(str(atts))
        file.write(f",{target}")
        file.write("\n")  # open a new line

    file.close()
