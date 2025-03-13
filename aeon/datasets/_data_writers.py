"""Dataset wrting functions."""

import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

from aeon.transformations.format import SlidingWindowTransformer, TrainTestTransformer

__all__ = [
    "write_to_ts_file",
    "write_to_tsf_file",
    "write_to_arff_file",
]


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


def write_to_tsf_file(
    df,
    full_file_path,
    metadata,
    value_column_name="series_value",
    attributes_types=None,
    missing_val_symbol="?",
):
    """
    Save a pandas DataFrame in TSF format.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be saved. It is assumed that one column contains the series
        (by default, named "series_value") and all other columns are series attributes.
    full_file_path : str
        The full path (including file name) where the TSF file will be saved.
    metadata : dict
        A dictionary containing metadata for the forecasting problem. It must
        include the following keys:
          - "frequency" (str)
          - "forecast_horizon" (int)
          - "contain_missing_values" (bool)
          - "contain_equal_length" (bool)
    value_column_name : str, optional (default="series_value")
        The name of the column that contains the time series values.
    attributes_types : dict, optional
        A dictionary mapping attribute column names to their TSF type
        (one of "numeric", "string", "date").
        If not provided, the type is inferred from the DataFrame dtypes as follows:
          - numeric dtypes -> "numeric"
          - datetime dtypes -> "date"
          - all others -> "string"
    missing_val_symbol : str, optional (default="?")
        The symbol to be used in the file to represent missing values in the series.

    Raises
    ------
    Exception
        If any required metadata or a series or attribute value is missing.
    """
    # Validate metadata keys
    required_meta = [
        "frequency",
        "forecast_horizon",
        "contain_missing_values",
        "contain_equal_length",
    ]
    for key in required_meta:
        if key not in metadata:
            raise AttributeError(f"Missing metadata entry: {key}")

    # Determine attribute columns (all columns except the series column)
    attribute_columns = [col for col in df.columns if col != value_column_name]

    # If no attributes are present, warn the user.
    if not attribute_columns:
        raise AttributeError(
            "The DataFrame must contain at least one \
                             attribute column besides the series column."
        )

    # Determine attribute types if not provided.
    # For each attribute, assign a type:
    # - numeric dtypes -> "numeric"
    # - datetime dtypes -> "date" (and will be formatted as "%Y-%m-%d %H-%M-%S")
    # - all others -> "string"
    if attributes_types is None:
        attributes_types = {}
        for col in attribute_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                attributes_types[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                attributes_types[col] = "date"
            else:
                attributes_types[col] = "string"
    else:
        # Ensure that a type is provided for each attribute column
        for col in attribute_columns:
            if col not in attributes_types:
                raise ValueError(
                    f"Attribute type for column '{col}' is \
                                 missing in attributes_types."
                )

    # Build header lines for the TSF file.
    header_lines = []
    # First, write the attribute lines (order matters!)
    for col in attribute_columns:
        att_type = attributes_types[col]
        if att_type not in {"numeric", "string", "date"}:
            raise ValueError(
                f"Unsupported attribute type '{att_type}' for column '{col}'."
            )
        header_lines.append(f"@attribute {col} {att_type}")

    # Now add the metadata lines. (The order here is flexible,
    # but must appear before @data.)
    header_lines.append(f"@frequency {metadata['frequency']}")
    header_lines.append(f"@horizon {metadata['forecast_horizon']}")
    header_lines.append(
        f"@missing {'true' if metadata['contain_missing_values'] else 'false'}"
    )
    header_lines.append(
        f"@equallength {'true' if metadata['contain_equal_length'] else 'false'}"
    )

    # Add the data section tag.
    header_lines.append("@data")
    # Open file for writing using the same encoding as the loader.
    with open(full_file_path, "w", encoding="cp1252") as f:
        # Write header lines.
        for line in header_lines:
            f.write(line + "\n")

        # Process each row to write the data lines.
        for idx, row in df.iterrows():
            parts = []
            # Process each attribute value.
            for col in attribute_columns:
                val = row[col]
                col_type = attributes_types[col]
                if pd.isna(val):
                    raise ValueError(
                        f"Missing value in attribute column '{col}' at row {idx}."
                    )
                if col_type == "numeric":
                    try:
                        val_str = str(int(val))
                    except Exception as e:
                        raise ValueError(
                            f"Error converting value in column '{col}' \
                                         at row {idx} to integer: {e}"
                        ) from e
                elif col_type == "date":
                    # Ensure val is a datetime; if not, attempt conversion.
                    if not isinstance(val, datetime):
                        try:
                            val = pd.to_datetime(val)
                        except Exception as e:
                            raise ValueError(
                                f"Error converting value in column '{col}' \
                                             at row {idx} to datetime: {e}"
                            ) from e
                    val_str = val.strftime("%Y-%m-%d %H-%M-%S")
                elif col_type == "string":
                    val_str = str(val)
                else:
                    # Should not get here because we validated types earlier.
                    raise ValueError(
                        f"Unsupported attribute type '{col_type}' for column '{col}'."
                    )
                parts.append(val_str)

            # Process the series data from value_column_name.
            series_val = row[value_column_name]
            if not hasattr(series_val, "__iter__"):
                raise ValueError(
                    f"The series in column '{value_column_name}' \
                                 at row {idx} is not iterable."
                )

            series_str_parts = []
            for s in series_val:
                # Check for missing values in the series.
                if pd.isna(s):
                    series_str_parts.append(missing_val_symbol)
                else:
                    series_str_parts.append(str(s).removesuffix(".0"))
            # Join series values with commas.
            series_str = ",".join(series_str_parts)
            parts.append(series_str)

            # The data line consists of the attribute values and
            # then the series, separated by colons.
            line_data = ":".join(parts)
            f.write(line_data + "\n")


def _write_header(
    path,
    problem_name,
    univariate=True,
    equal_length=False,
    n_timepoints=-1,
    comment=None,
    regression=False,
    class_labels=None,
):
    if class_labels is not None and regression:
        raise ValueError("Cannot have class_labels true for a regression problem")
    # create path if it does not exist
    dir_path = os.path.join(path, "")
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Error trying to access {dir_path} in _write_header") from exc
    # create ts file in the path
    load_path = os.path.join(dir_path, problem_name)
    file = open(load_path, "w", encoding="utf-8")
    # write comment if any as a block at start of file
    if comment is not None:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")

    # Writes the header info for a ts file
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
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f" Wrong input data type {type(X)}. Convert to np.ndarray (n_cases, "
            f"n_channels, n_timepoints) if possible."
        )

    if len(X.shape) != 3 or X.shape[1] != 1:
        raise ValueError(
            f"X must be a 3D array with shape (n_cases, 1, n_timepoints), but "
            f"received {X.shape}"
        )

    with open(f"{path}/{problem_name}.arff", "w", encoding="utf-8") as file:

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


def write_regression_dataset(series, full_file_path, dataset_name):
    """Write a regression dataset to file."""
    train_series, test_series = TrainTestTransformer().fit_transform(series)
    X_train, Y_train, train_indices = SlidingWindowTransformer().fit_transform(
        train_series
    )
    X_test, Y_test, test_indices = SlidingWindowTransformer().fit_transform(test_series)
    write_to_ts_file(
        [[item] for item in X_train],
        full_file_path,
        Y_train,
        f"{dataset_name}_TRAIN",
        None,
        True,
    )
    write_to_ts_file(
        [[item] for item in X_test],
        full_file_path,
        Y_test,
        f"{dataset_name}_TEST",
        None,
        True,
    )


def write_forecasting_dataset(series, full_file_path, dataset_name):
    """Write a regression dataset to file."""
    train_series, test_series = TrainTestTransformer().fit_transform(series)
    train_df = pd.DataFrame(train_series)
    train_df.to_csv(f"{full_file_path}/{dataset_name}_TRAIN.csv")
    test_df = pd.DataFrame(test_series)
    test_df.to_csv(f"{full_file_path}/{dataset_name}_TEST.csv")
