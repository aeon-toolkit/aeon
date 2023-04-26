# -*- coding: utf-8 -*-
import itertools
import os
import textwrap

import numpy as np
import pandas as pd

from aeon.datasets._data_io import _write_header
from aeon.datatypes import check_is_scitype, convert_to
from aeon.transformations.base import BaseTransformer
from aeon.utils.validation.panel import check_X, check_X_y


def write_dataframe_to_tsfile(
    data,
    path,
    problem_name="sample_data",
    class_label=None,
    class_value_list=None,
    equal_length=False,
    series_length=-1,
    missing_values="NaN",
    comment=None,
    fold="",
):
    """
    Output a dataset in dataframe format to .ts file.

    Parameters
    ----------
    data: pandas dataframe
        The dataset in a dataframe to be written as a ts file
        which must be of the structure specified in the documentation
        examples/loading_data.ipynb.
        index |   dim_0   |   dim_1   |    ...    |  dim_c-1
           0  | pd.Series | pd.Series | pd.Series | pd.Series
           1  | pd.Series | pd.Series | pd.Series | pd.Series
          ... |    ...    |    ...    |    ...    |    ...
           n  | pd.Series | pd.Series | pd.Series | pd.Series
    path: str
        The full path to output the ts file to.
    problem_name: str, default="sample_data"
        The problemName to print in the header of the ts file and also the name of
        the file.
    class_label: list of str or None, default=None
        The problems class labels to show the possible class values for in the file
        header, optional.
    class_value_list: list, ndarray or None, default=None
        The class values for each case, optional.
    equal_length: bool, default=False
        Indicates whether each series is of equal length.
    series_length: int, default=-1
        Indicates the series length if they are of equal length.
    missing_values: str, default="NaN"
        Representation for missing values.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    fold: str or None, default=None
        Addon at the end of the filename, i.e. _TRAIN or _TEST.

    Returns
    -------
    None

    Notes
    -----
    This version currently does not support writing timestamp data.
    """
    # ensure data provided is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Data provided must be a DataFrame, passed a {type(data)}")
    data_valid, _, metadata = check_is_scitype(
        data, scitype="Panel", return_metadata=True
    )
    if not data_valid:
        raise ValueError("DataFrame provided is not a valid type")
    if equal_length != metadata["is_equal_length"]:
        raise ValueError(
            f"Argument passed for equal length = {equal_length} is not "
            f"true for the data passed"
        )
    if equal_length:
        # Convert to [cases][dimensions][length] numpy.
        data = convert_to(
            data,
            to_type="numpy3D",
            as_scitype="Panel",
            store_behaviour="freeze",
        )
        write_ndarray_to_tsfile(
            data,
            path,
            problem_name=problem_name,
            class_label=class_label,
            class_value_list=class_value_list,
            equal_length=equal_length,
            series_length=data.shape[2],
            missing_values=missing_values,
            comment=comment,
            fold=fold,
        )
    else:  # Write by iterating over dataframe
        if class_value_list is not None and class_label is None:
            class_label = np.unique(class_value_list)
        file = _write_header(
            path,
            problem_name,
            metadata["is_univariate"],
            metadata["is_equal_length"],
            series_length,
            class_label,
            fold,
            comment,
        )
        n_cases, n_dimensions = data.shape
        for i in range(0, n_cases):
            for j in range(0, n_dimensions):
                series = data.iloc[i, j]
                for k in range(0, series.size - 1):
                    file.write(f"{series[k]},")
                file.write(f"{series[series.size-1]}:")
            file.write(f"{class_value_list[i]}\n")


def write_ndarray_to_tsfile(
    data,
    path,
    problem_name="sample_data",
    class_label=None,
    class_value_list=None,
    equal_length=False,
    series_length=-1,
    missing_values="NaN",
    comment=None,
    fold="",
):
    """
    Output a dataset in ndarray format to .ts file.

    Parameters
    ----------
    data: pandas dataframe
        The dataset in a 3d ndarray to be written as a ts file
        which must be of the structure specified in the documentation
        examples/loading_data.ipynb.
        (n_instances, n_columns, n_timepoints)
    path: str
        The full path to output the ts file to.
    problem_name: str, default="sample_data"
        The problemName to print in the header of the ts file and also the name of
        the file.
    class_label: list of str or None, default=None
        The problems class labels to show the possible class values for in the file
        header.
    class_value_list: list, ndarray or None, default=None
        The class values for each case, optional.
    equal_length: bool, default=False
        Indicates whether each series is of equal length.
    series_length: int, default=-1
        Indicates the series length if they are of equal length.
    missing_values: str, default="NaN"
        Representation for missing values.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    fold: str or None, default=None
        Addon at the end of the filename, i.e. _TRAIN or _TEST.

    Returns
    -------
    None

    Notes
    -----
    This version currently does not support writing timestamp data.
    """
    # ensure data provided is a ndarray
    if not isinstance(data, np.ndarray):
        raise ValueError("Data provided must be a ndarray")
    if class_value_list is not None:
        data, class_value_list = check_X_y(data, class_value_list)
    else:
        data = check_X(data)
    univariate = data.shape[1] == 1
    if class_value_list is not None and class_label is None:
        class_label = np.unique(class_value_list)
    elif class_value_list is None:
        class_value_list = []
    # ensure number of cases is same as the class value list
    if len(data) != len(class_value_list) and len(class_value_list) > 0:
        raise IndexError(
            "The number of cases is not the same as the number of given class values"
        )
    if equal_length and series_length == -1:
        raise ValueError(
            "Please specify the series length for equal length time series data."
        )
    if fold is None:
        fold = ""
    file = _write_header(
        path,
        problem_name,
        univariate,
        equal_length,
        series_length,
        class_label,
        fold,
        comment,
    )
    # begin writing the core data for each case
    # which are the series and the class value list if there is any
    for case, value in itertools.zip_longest(data, class_value_list):
        for dimension in case:
            # turn series into comma-separated row
            series = ",".join(
                [str(num) if not np.isnan(num) else missing_values for num in dimension]
            )
            file.write(str(series))
            # continue with another dimension for multivariate case
            if not univariate:
                file.write(":")
        a = ":" if univariate else ""
        if value is not None:
            file.write(f"{a}{value}")  # write the case value if any
        elif class_label is not None:
            file.write(f"{a}{missing_values}")
        file.write("\n")  # open a new line
    file.close()


def write_panel_to_tsfile(
    data, path, target=None, problem_name="sample_data", header=None
):
    """Write an aeon multi-instance dataset to text file in .ts format.

    Write metadata and data stored in aeon compatible data set to file.
    A description of the ts format is in docs/api_reference/data_format.rst

    Parameters
    ----------
    data : Panel.
        dataset containing multiple time series instances, referred to as a Panel in
        aeon.
        Series can univariate, multivariate, equal or unequal length
    path : String.
        Location of the directory to write file
    target: None or ndarray, default = None
        Response variable, discrete for classification, continuous for regression
        None if clustering.
    problem_name : String, default = "sample_data"
        The file is written to <path>/<problem_name>/<problem_name>.ts
    header: String, default = None
        Optional text at the top of the file that is ignored when loading.
    """
    data_valid, _, data_metadata = check_is_scitype(
        data, scitype="Panel", return_metadata=True
    )
    if not data_valid:
        raise TypeError(" Wrong input data type ", type(data))
    if data_metadata["is_equal_length"]:
        # check class labels
        data = convert_to(
            data,
            to_type="numpy3D",
            as_scitype="Panel",
            store_behaviour="freeze",
        )
        series_length = data.shape[2]
        write_ndarray_to_tsfile(
            data,
            path,
            problem_name=problem_name,
            class_value_list=target,
            equal_length=True,
            series_length=series_length,
            comment=header,
        )
    else:
        write_dataframe_to_tsfile(
            data,
            path,
            problem_name=problem_name,
            class_value_list=target,
            equal_length=False,
            comment=header,
        )


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


def write_tabular_transformation_to_arff(
    data,
    transformation,
    path,
    problem_name="sample_data",
    class_label=None,
    class_value_list=None,
    comment=None,
    fold="",
    fit_transform=True,
):
    """
    Transform a dataset using a tabular transformer and write the result to a arff file.

    Parameters
    ----------
    data: pandas dataframe or 3d numpy array
        The dataset to build the transformation with which must be of the structure
        specified in the documentation examples/loading_data.ipynb.
    transformation: BaseTransformer
        Transformation use and to save to arff.
    path: str
        The full path to output the arff file to.
    problem_name: str, default="sample_data"
        The problemName to print in the header of the arff file and also the name of
        the file.
    class_label: list of str or None, default=None
        The problems class labels to show the possible class values for in the file
        header, optional.
    class_value_list: list, ndarray or None, default=None
        The class values for each case, optional.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    fold: str or None, default=None
        Addon at the end of the filename, i.e. _TRAIN or _TEST.
    fit_transform: bool, default=True
        Whether to fit the transformer prior to calling transform.

    Returns
    -------
    None
    """
    # ensure transformation provided is a transformer
    if not isinstance(transformation, BaseTransformer):
        raise ValueError("Transformation must be a BaseTransformer")
    if fit_transform:
        data = transformation.fit_transform(data, class_value_list)
    else:
        data = transformation.transform(data, class_value_list)
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    if class_value_list is not None and class_label is None:
        class_label = np.unique(class_value_list)
    elif class_value_list is None:
        class_value_list = []
    # ensure number of cases is same as the class value list
    if len(data) != len(class_value_list) and len(class_value_list) > 0:
        raise IndexError(
            "The number of cases is not the same as the number of given class values"
        )
    if fold is None:
        fold = ""
    # create path if not exist
    dirt = f"{str(path)}/{str(problem_name)}-{type(transformation).__name__}/"
    try:
        os.makedirs(dirt)
    except os.error:
        pass  # raises os.error if path already exists
    # create arff file in the path
    file = open(
        f"{dirt}{str(problem_name)}-{type(transformation).__name__}{fold}.arff", "w"
    )
    # write comment if any as a block at start of file
    if comment is not None:
        file.write("\n% ".join(textwrap.wrap("% " + comment)))
        file.write("\n")
    # begin writing header information
    file.write(f"@Relation {problem_name}\n")
    # write each attribute
    for i in range(data.shape[1]):
        file.write(f"@attribute att{str(i)} numeric\n")
    # write class attribute if it exists
    if class_label is not None:
        comma_separated_class_label = ",".join(str(label) for label in class_label)
        file.write(f"@attribute target {{{comma_separated_class_label}}}\n")
    file.write("@data\n")
    for case, value in itertools.zip_longest(data, class_value_list):
        # turn attributes into comma-separated row
        atts = ",".join([str(num) if not np.isnan(num) else "?" for num in case])
        file.write(str(atts))
        if value is not None:
            file.write(f",{value}")  # write the case value if any
        elif class_label is not None:
            file.write(",?")
        file.write("\n")  # open a new line
    file.close()
