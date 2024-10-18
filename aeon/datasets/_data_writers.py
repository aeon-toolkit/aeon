import os
import textwrap

import numpy as np

__all__ = ["write_to_tsfile", "write_to_arff_file"]


def write_to_tsfile(
    X, path, y=None, problem_name="sample_data.ts", header=None, regression=False
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
        Indicate if this is a regression problem, so it is correcty specified in
        the header since there is no definite way of inferring this from y
    """
    if not (isinstance(X, np.ndarray) or isinstance(X, list)):
        raise TypeError(
            f" Wrong input data type {type(X)} convert to np.ndarray ("
            f"n_cases, n_channels,n_timepoints) if equal length or list "
            f"of [n_cases] np.ndarray shape (n_channels, n_timepoints) if unequal"
        )

    # See if passed file name contains .ts extension or not
    split = problem_name.split(".")
    if split[-1] != "ts":
        problem_name = problem_name + ".ts"
    _write_data_to_tsfile(
        X, path, problem_name, y=y, comment=header, regression=regression
    )


def _write_data_to_tsfile(
    X,
    path,
    problem_name,
    y=None,
    missing_values="NaN",
    comment=None,
    suffix=None,
    regression=False,
):
    """Output a dataset to .ts texfile format.

    Automatically adds the .ts suffix if not the suffix to problem_name.

    Parameters
    ----------
    X: Union[list, np.ndarray]
        time series collection, either a 3d ndarray  (n_cases, n_channels,
        n_timepoints) or a list of [n_cases] 2d numpy arrays (possibly variable
        length)
    path: str
        The full path to output the ts file to.
    problem_name: str
        The problemName to print in the header of the ts file and also the name of
        the file.
    y: list, ndarray or None, default=None
        The class values for each case, optional.
    missing_values: str, default="NaN"
        Representation for missing values.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    suffix: str or None, default=None
        Addon at the end of the filename before the file extension, i.e. _TRAIN or
        _TEST

    Returns
    -------
    None

    Notes
    -----
    This version currently does not support writing timestamp data.
    """
    # ensure data provided is a ndarray. This is redundant here as already checked
    if not isinstance(X, np.ndarray) and not isinstance(X, list):
        raise ValueError("Data provided must be a ndarray or a list")
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
        comment=comment,
        regression=regression,
        suffix=suffix,
        extension=None,
    )
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
    suffix=None,
    extension=None,
):
    if class_labels is not None and regression:
        raise ValueError("Cannot have class_labels true for a regression problem")
    # create path if it does not exist
    dir = f"{str(path)}/"
    try:
        os.makedirs(dir, exist_ok=True)
    except OSError:
        raise ValueError(f"Error trying to access {dir} in _write_header")
    # create ts file in the path
    load_path = f"{dir}{str(problem_name)}"
    if suffix is not None:
        load_path = load_path + suffix
    if extension is not None:
        load_path = load_path + extension
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
        if regression:  # or if a regresssion problem, write target label
            file.write("@targetlabel true\n")
    file.write("@data\n")
    return file


def write_to_arff_file(
    X,
    y,
    path,
    problem_name="sample_data",
    header=None,
    suffix="",
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
    suffix: str or None, default=""
        Addon at the end of the filename before the file extension, i.e. _TRAIN or
        _TEST.

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

    file = open(f"{path}/{problem_name}{suffix}.arff", "w")

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
