# -*- coding: utf-8 -*-
import os
import textwrap

import numpy as np
import pandas as pd

__all__ = ["write_to_tsfile", "write_results_to_uea_format"]


def write_to_tsfile(
    X, path, y=None, problem_name="sample_data.ts", header=None, regression=False
):
    """Write an aeon collection of time series to text file in .ts format.

    Write metadata and data stored in aeon compatible data set to file.
    A description of the ts format is in examples/load_data.ipynb.

    Note that this file is structured to still support the

    Parameters
    ----------
    X : np.ndarray (n_cases, n_channels, series_length) or list of np.ndarray[
    n_cases] or pd.DataFrame with (n_cases,n_channels), each cell a pd.Series
        Collection of time series: univariate, multivariate, equal or unequal length.
    path : string.
        Location of the directory to write file
    y: None or ndarray, default = None
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
    if not (
        isinstance(X, np.ndarray) or isinstance(X, list) or isinstance(pd.DataFrame)
    ):
        raise TypeError(
            f" Wrong input data type {type(X)} convert to np.ndarray ("
            f"n_cases, n_channels,n_timepoints) if equal length or list "
            f"of [n_cases] np.ndarray shape (n_channels, n_timepoints) if unequal"
        )
    # See if passed file name contains .ts extension or not
    split = problem_name.split(".")
    if split[-1] != "ts":
        problem_name = problem_name + ".ts"

    if isinstance(X, np.ndarray) or isinstance(X, list):
        _write_data_to_tsfile(X, path, problem_name, y=y, regression=regression)
    else:
        _write_dataframe_to_tsfile(
            X,
            path,
            problem_name=problem_name,
            y=y,
            equal_length=False,
            comment=header,
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
    # ensure data provided is a ndarray
    if not isinstance(X, np.ndarray) and not isinstance(X, list):
        raise ValueError("Data provided must be a ndarray or a list")
    class_labels = None
    if y is not None and not regression:
        # ensure number of cases is same as the class value list
        if len(X) != len(y):
            raise IndexError(
                "The number of cases is not the same as the number of  class values"
            )
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
    series_length = -1
    if equal_length:
        series_length = len(X[0][0])
    file = _write_header(
        path,
        problem_name,
        univariate=univariate,
        equal_length=equal_length,
        series_length=series_length,
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


def write_results_to_uea_format(
    estimator_name,
    dataset_name,
    y_pred,
    output_path,
    full_path=True,
    y_true=None,
    predicted_probs=None,
    split="TEST",
    resample_seed=0,
    timing_type="N/A",
    first_line_comment=None,
    second_line="No Parameter Info",
    third_line="N/A",
):
    """Write the predictions for an experiment in the standard format used by aeon.

    Parameters
    ----------
    estimator_name : str,
        Name of the object that made the predictions, written to file and can
        deterimine file structure of output_root is True
    dataset_name : str
        name of the problem the classifier was built on
    y_pred : np.array
        predicted values
    output_path : str
        Path where to put results. Either a root path, or a full path
    full_path : boolean, default = True
        If False, then the standard file structure is created. If false, results are
        written directly to the directory passed in output_path
    y_true : np.array, default = None
        Actual values, written to file with the predicted values if present
    predicted_probs :  np.ndarray, default = None
        Estimated class probabilities. If passed, these are written after the
        predicted values. Regressors should not pass anything
    split : str, default = "TEST"
        Either TRAIN or TEST, depending on the results, influences file name.
    resample_seed : int, default = 0
        Indicates what data
    timing_type : str or None, default = None
        The format used for timings in the file, i.e. Seconds, Milliseconds, Nanoseconds
    first_line_comment : str or None, default = None
        Optional comment appended to the end of the first line
    second_line : str
        unstructured, used for predictor parameters
    third_line : str
        summary performance information (see comment below)
    """
    if len(y_true) != len(y_pred):
        raise IndexError(
            "The number of predicted values is not the same as the "
            "number of actual class values"
        )
    # If the full directory path is not passed, make the standard structure
    if not full_path:
        output_path = f"{output_path}/{estimator_name}/Predictions/{dataset_name}/"
    try:
        os.makedirs(output_path)
    except os.error:
        pass  # raises os.error if path already exists, so just ignore this

    if split == "TRAIN" or split == "train":
        train_or_test = "train"
    elif split == "TEST" or split == "test":
        train_or_test = "test"
    else:
        raise ValueError("Unknown 'split' value - should be TRAIN/train or TEST/test")
    file = open(f"{output_path}/{train_or_test}Resample{resample_seed}.csv", "w")
    # the first line of the output file is in the form of:
    # <classifierName>,<datasetName>,<train/test>
    first_line = f"{dataset_name},{estimator_name},{train_or_test},{resample_seed}"
    if timing_type is not None:
        first_line += "," + timing_type
    if first_line_comment is not None:
        first_line += "," + first_line_comment
    file.write(first_line + "\n")
    # the second line of the output is free form and estimator-specific; usually this
    # will record info such as build time, paramater options used, any constituent model
    # names for ensembles, etc.
    file.write(str(second_line) + "\n")
    # the third line of the file is the accuracy (should be between 0 and 1
    # inclusive). If this is a train output file then it will be a training estimate
    # of the classifier on the training data only (e.g. 10-fold cv, leave-one-out cv,
    # etc.). If this is a test output file, it should be the output of the estimator
    # on the test data (likely trained on the training data for a-priori parameter
    # optimisation)
    file.write(str(third_line) + "\n")
    # from line 4 onwards each line should include the actual and predicted class
    # labels (comma-separated). If present, for each case, the probabilities of
    # predicting every class value for this case should also be appended to the line (
    # a space is also included between the predicted value and the predict_proba). E.g.:
    #
    # if predict_proba data IS provided for case i:
    #   y_true[i], y_pred[i],,prob_class_0[i],
    #   prob_class_1[i],...,prob_class_c[i]
    #
    # if predict_proba data IS NOT provided for case i:
    #   y_true[i], y_pred[i]
    # If y_true is None (if clustering), y_true[i] is replaced with ? to indicate
    # missing
    if y_true is None:
        for i in range(0, len(y_pred)):
            file.write("?," + str(y_pred[i]))
            if predicted_probs is not None:
                file.write(",")
                for j in predicted_probs[i]:
                    file.write("," + str(j))
            file.write("\n")
    else:
        for i in range(0, len(y_pred)):
            file.write(str(y_true[i]) + "," + str(y_pred[i]))
            if predicted_probs is not None:
                file.write(",")
                for j in predicted_probs[i]:
                    file.write("," + str(j))
            file.write("\n")
    file.close()


def _write_header(
    path,
    problem_name,
    univariate=True,
    equal_length=False,
    series_length=-1,
    comment=None,
    regression=False,
    class_labels=None,
    suffix=None,
    extension=None,
):
    if class_labels is not None and regression:
        raise ValueError(
            "Cannot have class_labels and targetlabel. If the problem "
            "is classification, add class_labels. If regression, "
            "set targetlabel to true."
        )
    # create path if it does not exist
    dir = f"{str(path)}/"
    try:
        os.makedirs(dir, exist_ok=True)
    except os.error:
        raise ValueError(f"Error trying to access {dir} in _write_header")
    # create ts file in the path
    load_path = f"{dir}{str(problem_name)}"
    if suffix is not None:
        load_path = load_path + suffix
    if extension is not None:
        load_path = load_path + suffix + extension
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
    if series_length > 0 and equal_length:
        file.write(f"@seriesLength {series_length}\n")
    # write class labels line
    if class_labels is not None:
        space_separated_class_label = " ".join(str(label) for label in class_labels)
        file.write(f"@classLabel true {space_separated_class_label}\n")
    else:
        file.write("@classLabel false\n")
        if regression:
            file.write("@targetlabel true\n")

    # or if a regresssion problem, write target label
    file.write("@data\n")
    return file


def _write_dataframe_to_tsfile(
    X,
    path,
    problem_name="sample_data",
    y=None,
    comment=None,
):
    # ensure data provided is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"Data provided must be a DataFrame, passed a {type(X)}")
    # See if passed file name contains .ts extension or not
    split = problem_name.split(".")
    if split[-1] != "ts":
        problem_name = problem_name + ".ts"
    class_labels = None
    if y is not None:
        class_labels = np.unique(y)
    file = _write_header(
        path,
        problem_name,
        comment=comment,
        class_labels=class_labels,
    )
    n_cases, n_dimensions = X.shape
    for i in range(0, n_cases):
        for j in range(0, n_dimensions):
            series = X.iloc[i, j]
            for k in range(0, series.size - 1):
                file.write(f"{series[k]},")
            file.write(f"{series[series.size-1]}:")
        file.write(f"{y[i]}\n")
    file.close()
