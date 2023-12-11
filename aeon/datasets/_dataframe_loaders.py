"""Legacy functions that load collections of time series into nested dataframes."""

__author__ = [
    "Emiliathewolf",
    "TonyBagnall",
    "jasonlines",
    "achieveordie",
]

__all__ = [
    "load_from_tsfile_to_dataframe",
    "load_from_arff_to_dataframe",
    "load_from_ucr_tsv_to_dataframe",
]
import os
from datetime import datetime
from distutils.util import strtobool

import pandas as pd

import aeon
from aeon.datasets._data_generators import _convert_tsf_to_hierarchical
from aeon.datasets._data_loaders import load_from_arff_file, load_from_tsfile
from aeon.utils.validation.collection import convert_collection

DIRNAME = "data"
MODULE = os.path.join(os.path.dirname(aeon.__file__), "datasets")


def load_from_tsfile_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a nested pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name : str
        The full pathname of the .ts file to read.
    replace_missing_vals_with : str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    data : DataFrame
        Time series data in nested pd.DataFrame format.
    y : np.ndarray
        Target variable.
    """
    X, y = load_from_tsfile(full_file_path_and_name, replace_missing_vals_with)
    # Convert
    X = convert_collection(X, output_type="nested_univ")
    return X, y


def load_from_arff_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
):
    """Load data from a .arff file into a nested pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name : str
        The full pathname of the .ts file to read.
    replace_missing_vals_with : str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    data : DataFrame
        Time series data in nested pd.DataFrame format.
    y : np.ndarray
        Target variable.
    """
    X, y = load_from_arff_file(full_file_path_and_name, replace_missing_vals_with)
    # Convert
    X = convert_collection(X, output_type="nested_univ")
    return X, y


def load_from_ucr_tsv_to_dataframe(
    full_file_path_and_name, return_separate_X_and_y=True
):
    """Load data from a .tsv file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .tsv file to read.
    return_separate_X_and_y: bool
        true then X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    df = pd.read_csv(full_file_path_and_name, sep="\t", header=None)
    y = df.pop(0).values
    df.columns -= 1
    X = pd.DataFrame()
    X["var_0"] = [pd.Series(df.iloc[x, :]) for x in range(len(df))]
    if return_separate_X_and_y is True:
        return X, y
    X["class_val"] = y
    return X


def load_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
    return_type="tsf_default",
):
    """
    Convert the contents in a .tsf file into a dataframe.

    This code was extracted from
    https://github.com/rakshitha123/TSForecasting/blob
    /master/utils/data_loader.py.

    Parameters
    ----------
    full_file_path_and_name: str
        The full path to the .tsf file.
    replace_missing_vals_with: str, default="NAN"
        A term to indicate the missing values in series in the returning dataframe.
    value_column_name: str, default="series_value"
        Any name that is preferred to have as the name of the column containing series
        values in the returning dataframe.
    return_type : str - "pd_multiindex_hier" or "tsf_default" (default)
        - "tsf_default" = container that faithfully mirrors tsf format from the original
            implementation in: https://github.com/rakshitha123/TSForecasting/
            blob/master/utils/data_loader.py.

    Returns
    -------
    loaded_data: pd.DataFrame
        The converted dataframe containing the time series.
    metadata: dict
        The metadata for the forecasting problem. The dictionary keys are:
        "frequency", "forecast_horizon", "contain_missing_values",
        "contain_equal_length"
    """
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. "
                                "Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. "
                            "Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set "
                                "of comma separated numeric values."
                                "At least one numeric value should be there "
                                "in a series. "
                                "Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. "
                                "A given series should contains a set "
                                "of comma separated numeric values."
                                "At least one numeric value should be there "
                                "in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                # Currently, the code supports only
                                # numeric, string and date types.
                                # Extend this as required.
                                raise Exception("Invalid attribute type.")

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        # metadata dict
        metadata = dict(
            zip(
                (
                    "frequency",
                    "forecast_horizon",
                    "contain_missing_values",
                    "contain_equal_length",
                ),
                (
                    frequency,
                    forecast_horizon,
                    contain_missing_values,
                    contain_equal_length,
                ),
            )
        )

        if return_type != "tsf_default":
            loaded_data = _convert_tsf_to_hierarchical(
                loaded_data, metadata, value_column_name=value_column_name
            )
        return loaded_data, metadata
