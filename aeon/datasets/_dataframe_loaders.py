"""Legacy functions that load collections of time series into nested dataframes."""

__maintainer__ = []
__all__ = [
    "load_from_tsfile_to_dataframe",
    "load_from_arff_to_dataframe",
    "load_from_ucr_tsv_to_dataframe",
]
import os

import pandas as pd

import aeon
from aeon.datasets._data_loaders import load_from_arff_file, load_from_tsfile
from aeon.utils.conversion import convert_collection

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
