"""Dataset loading functions."""

from typing import Optional

__all__ = [  # Load functions
    "load_from_ts_file",
    "load_from_tsf_file",
    "load_from_arff_file",
    "load_from_tsv_file",
    "load_classification",
    "load_forecasting",
    "load_regression",
    "download_all_regression",
    "get_dataset_meta_data",
]

import glob
import os
import re
import shutil
import socket
import tempfile
import urllib
import zipfile
from datetime import datetime
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve

import numpy as np
import pandas as pd

import aeon
from aeon.datasets.dataset_collections import (
    get_downloaded_tsc_tsr_datasets,
    get_downloaded_tsf_datasets,
)
from aeon.datasets.tsc_datasets import tsc_zenodo
from aeon.datasets.tser_datasets import tser_monash, tser_soton
from aeon.utils.conversion import convert_collection

DIRNAME = "data"
MODULE = os.path.join(os.path.dirname(aeon.__file__), "datasets")

CONNECTION_ERRORS = (
    HTTPError,
    URLError,
    RemoteDisconnected,
    IncompleteRead,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
)


# Return appropriate return_type in case an alias was used
def _alias_datatype_check(return_type):
    if return_type in ["numpy2d", "np2d", "np2D", "numpyflat"]:
        return_type = "numpy2D"
    if return_type in ["numpy3d", "np3d", "np3D"]:
        return_type = "numpy3D"
    return return_type


def _load_header_info(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : stream.
        input file to read header from, assumed to be just opened

    Returns
    -------
    meta_data : dict.
        dictionary with the data characteristics stored in the header.
    """
    meta_data = {
        "problemname": "none",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    boolean_keys = ["timestamps", "missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        line = re.sub(r"\s+", " ", line)
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise OSError("data tag should not have an associated value")
                return meta_data
            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise OSError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data[key] = True
                    elif tokens[1] == "false":
                        meta_data[key] = False
                elif key == "problemname":
                    meta_data[key] = tokens[1]
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data["classlabel"] = True
                        if token_len == 2:
                            raise OSError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise OSError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data


def _get_channel_strings(line, target=True, missing="NaN"):
    """Split a string with timestamps into separate csv strings."""
    channel_strings = re.sub(r"\s", "", line)
    channel_strings = channel_strings.split("):")
    c = len(channel_strings)
    if target:
        c = c - 1
    for i in range(c):
        channel_strings[i] = channel_strings[i] + ")"
        numbers = re.findall(r"\d+\.\d+|" + missing, channel_strings[i])
        channel_strings[i] = ",".join(numbers)
    return channel_strings


def _load_data(file, meta_data, replace_missing_vals_with="NaN"):
    """Load data from a file with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : stream, input file to read data from, assume no comments or header info
    meta_data : dict.
        with meta data in the file header loaded with _load_header_info

    Returns
    -------
    data: list[np.ndarray].
        list of numpy arrays of floats: the time series
    y_values : np.ndarray.
        numpy array of strings: the class/target variable values
    meta_data :  dict.
        dictionary of characteristics enhanced with number of channels and series length
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    n_timepoints = 0
    y_values = []
    target = False
    if meta_data["classlabel"] or meta_data["targetlabel"]:
        target = True
    for line in file:
        line = line.strip().lower()
        line = line.replace("nan", replace_missing_vals_with)
        line = line.replace("?", replace_missing_vals_with)
        if "timestamps" in meta_data and meta_data["timestamps"]:
            channels = _get_channel_strings(line, target, replace_missing_vals_with)
        else:
            channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if target:
            current_channels -= 1
        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels = current_channels
            if meta_data["equallength"]:
                n_timepoints = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise OSError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"]:
                if current_channels > 1:
                    raise OSError(
                        f"Seen {current_channels} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
        if meta_data["equallength"]:
            current_length = n_timepoints
        else:
            current_length = len(channels[0].split(","))
        np_case = np.zeros(shape=(n_channels, current_length))
        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length:
                equal_length = meta_data["equallength"]
                raise OSError(
                    f"channel {i} in case {n_cases} has a different number of "
                    f"observations to the other channels. "
                    f"Saw {current_length} in the first channel but"
                    f" {len(data_series)} in the channel {i}. The meta data "
                    f"specifies equal length == {equal_length}. But even if series "
                    f"length are unequal, all channels for a single case must be the "
                    f"same length"
                )
            np_case[i] = np.array(data_series)
        data.append(np_case)
        if target:
            y_values.append(channels[n_channels])
    if meta_data["equallength"]:
        data = np.array(data)
    return data, np.asarray(y_values), meta_data


def load_from_ts_file(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
        full path of the file to load, .ts extension is assumed.
    replace_missing_vals_with : string, default="NaN"
        issing values in the file are replaces with this value
    return_meta_data : boolean, default=False
        return a dictionary with the meta data loaded from the file
    return_type : string, default = "auto"
        data type to convert to.
        If "auto", returns numpy3D for equal length and list of numpy2D for unequal.
        If "numpy2D", will squash a univariate equal length into a numpy2D (n_cases,
        n_timepoints). Other options are available but not supported medium term.

    Returns
    -------
    data: np.ndarray or list of np.ndarray
        time series data, np.ndarray (n_cases, n_channels, n_timepoints) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : np.ndarray of string or int
        target variable
    meta_data : dict, optional.
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # split the file path into the root and the extension
    root, ext = os.path.splitext(full_file_path_and_name)
    # Append .ts if no extension if found
    if not ext:
        full_file_path_and_name = root + ".ts"
    # Open file
    with open(full_file_path_and_name, encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        # load into list of numpy
        data, y, meta_data = _load_data(file, meta_data)

    # if equal load to 3D numpy
    if meta_data["equallength"]:
        data = np.array(data)
        if return_type == "numpy2D" and meta_data["univariate"]:
            data = data.squeeze()
    # If regression problem, convert y to float
    if meta_data["targetlabel"]:
        y = y.astype(float)
    if return_meta_data:
        return data, y, meta_data
    return data, y


def _load_saved_dataset(
    name,
    split=None,
    return_type=None,
    local_module=MODULE,
    local_dirname=DIRNAME,
    return_meta=False,
    dir_name=None,
):
    """Load baked in time series classification datasets (helper function).

    Loads data from the provided files from aeon/datasets/data only.

    Parameters
    ----------
    name : str
        Base problem file name.
    split: None or {"TRAIN", "TEST"}, default=None
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances into a single data structure.
    return_data_type : str, default = None
        "numpy3D"/"numpy3d"/"np3D": recommended for equal length series, "np-list"
        for unequal length series that cannot be stored in numpy arrays.
        "numpy2D"/"numpy2d"/"np2d": can be used for univariate equal length series,
        although we recommend numpy3d, because some transformers do not work with
        numpy2d. If None will load 3D numpy or list of numpy.
    local_module: default = os.path.dirname(__file__),
    local_dirname: str, default = "data"
    return_meta: bool, default = False
        Dictionary of characteristics "problemname" (string), booleans: "timestamps",
        "missing", "univariate", "equallength", "classlabel", "targetlabel" and
        "class_values": [].
    dir_name: str, default = None
        Directory in local_dirname containing the problem file. If None, dir_name = name

    Returns
    -------
    X: Data stored in specified `return_type`
        The time series collection for the problem.
    y: 1D numpy array of length len(X)
     The class labels for each time series instance in X.

    meta: dict
        Dictionary of data characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and list: "class_values",

    Raises
    ------
    Raise ValueError if the requested return type is not supported
    """
    if isinstance(split, str):
        split = split.upper()
    # This is also called in load_from_ts_file, but we need the value here and it
    # is required in load_from_ts_file since it is public
    return_type = _alias_datatype_check(return_type)
    if dir_name is None:
        dir_name = name
    if local_dirname is not None:
        local_module = os.path.join(local_module, local_dirname)
    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        abspath = os.path.join(local_module, dir_name, fname)
        X, y, meta_data = load_from_ts_file(abspath, return_meta_data=True)
    # if split is None, load both train and test set
    elif split is None:
        fname = name + "_TRAIN.ts"
        abspath = os.path.join(local_module, dir_name, fname)
        X_train, y_train, meta_data = load_from_ts_file(abspath, return_meta_data=True)

        fname = name + "_TEST.ts"
        abspath = os.path.join(local_module, dir_name, fname)
        X_test, y_test, meta_data_test = load_from_ts_file(
            abspath, return_meta_data=True
        )
        if meta_data["equallength"]:
            X = np.concatenate([X_train, X_test])
        else:
            X = X_train + X_test
        y = np.concatenate([y_train, y_test])
    else:
        raise ValueError("Invalid `split` value =", split)
    if return_type is not None:
        X = convert_collection(X, return_type)
    if return_meta:
        return X, y, meta_data
    else:
        return X, y


def download_dataset(name, save_path=None):
    """

    Download a dataset from the timeseriesclassification.com website.

    Parameters
    ----------
    name : string,
            name of the dataset to download

    safe_path : string, optional (default: None)
            Path to the directory where the dataset is downloaded into.

    Returns
    -------
    if successful, string containing the path of the saved file

    Raises
    ------
    Raise URLError or HTTPError if the website is not accessible,
    ValueError if a dataset name that does not exist on the repo
    is given.
    """
    if save_path is None:
        save_path = os.path.join(MODULE, "local_data")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if name not in get_downloaded_tsc_tsr_datasets(
        save_path
    ) or name not in get_downloaded_tsf_datasets(save_path):
        # Dataset is not already present in the datasets directory provided.
        # If it is not there, download it.
        url = f"https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
        try:
            _download_and_extract(url, extract_path=save_path)
        except zipfile.BadZipFile as e:
            raise ValueError(
                f"Invalid dataset name ={name} is available on extract path ="
                f" {save_path} or https://timeseriesclassification.com/ but it is not "
                f"correctly formatted.",
            ) from e

    return os.path.join(save_path, name)


def _download_and_extract(url, extract_path=None):
    """
    Download and unzip datasets (helper function).

    This code was modified from
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L28

    Parameters
    ----------
    url : string
        Url pointing to file to download
    extract_path : string, optional (default: None)
        path to extract downloaded zip to, None defaults
        to aeon/datasets/data

    Returns
    -------
    extract_path : string or None
        if successful, string containing the path of the extracted file, None
        if it wasn't successful
    """
    file_name = os.path.basename(url)
    dl_dir = tempfile.mkdtemp()
    zip_file_name = os.path.join(dl_dir, file_name)
    #    urlretrieve(url, zip_file_name)

    # Using urlopen instead of urlretrieve
    with urlopen(url, timeout=60) as response:
        with open(zip_file_name, "wb") as out_file:
            out_file.write(response.read())
    if extract_path is None:
        extract_path = os.path.join(MODULE, "local_data/%s/" % file_name.split(".")[0])
    else:
        extract_path = os.path.join(extract_path, "%s/" % file_name.split(".")[0])

    try:
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        zipfile.ZipFile(zip_file_name, "r").extractall(extract_path)
        shutil.rmtree(dl_dir)
        return extract_path
    except zipfile.BadZipFile:
        shutil.rmtree(dl_dir)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        raise zipfile.BadZipFile(
            "Could not unzip dataset. Please make sure the URL is valid."
        )


def _load_tsc_dataset(
    name, split, return_type=None, extract_path=None, return_meta=False
):
    """Load time series classification datasets (helper function).

    Parameters
    ----------
    name : string, file name to load from
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).
    return_data_type : str, optional, default = None
        "numpy3D"/"numpy3d"/"np3D": recommended for equal length series
        "numpy2D"/"numpy2d"/"np2d": can be used for univariate equal length series,
        although we recommend numpy3d, because some transformers do not work with
        numpy2d.
        "np-list": for unequal length series that cannot be stored in numpy arrays
        if None returns either numpy3D for equal length or  "np-list" for unequal
    extract_path : optional (default = None)
        Path of the location for the data file. If none, data is written to
        os.path.dirname(__file__)/data/

    Returns
    -------
    X: Data stored in specified `return_type`
        The time series data for the problem, with n instances.
    y: 1D numpy array of length len(X)
        The class labels for each time series instance in X.

    Raises
    ------
    Raise ValueException if the requested return type is not supported
    """
    # Allow user to have non standard extract path
    if extract_path is not None:
        local_module = extract_path
        local_dirname = ""
    else:
        local_module = MODULE
        local_dirname = "data"

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    if name not in get_downloaded_tsc_tsr_datasets(extract_path):
        if extract_path is None:
            local_dirname = "local_data"
        if not os.path.exists(os.path.join(local_module, local_dirname)):
            os.makedirs(os.path.join(local_module, local_dirname))
        if name not in get_downloaded_tsc_tsr_datasets(
            os.path.join(local_module, local_dirname)
        ):
            # Dataset is not already present in the datasets directory provided.
            # If it is not there, download and install it.
            url = "https://timeseriesclassification.com/aeon-toolkit/%s.zip" % name
            # This also tests the validitiy of the URL, can't rely on the html
            # status code as it always returns 200
            try:
                _download_and_extract(
                    url,
                    extract_path=extract_path,
                )
            except zipfile.BadZipFile as e:
                raise ValueError(
                    f"Invalid dataset name ={name} is not available on extract path ="
                    f"{extract_path}. Nor is it available on {url}",
                ) from e

    return _load_saved_dataset(
        name,
        split=split,
        return_type=return_type,
        local_module=local_module,
        local_dirname=local_dirname,
        return_meta=return_meta,
    )


def load_from_arff_file(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
):
    """Load data from a classification/regression WEKA arff file to a 3D numpy array.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    data: np.ndarray
        time series data, np.ndarray (n_cases, n_channels, n_timepoints)
    y : np.ndarray of string or int
        target variable
    """
    instance_list = []
    class_val_list = []
    data_started = False
    is_multi_variate = False
    is_first_case = True
    n_cases = 0
    n_channels = 1
    with open(full_file_path_and_name, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                if (
                    is_multi_variate is False
                    and "@attribute" in line.lower()
                    and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue
                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        line, class_val = line.split("',")
                        class_val_list.append(class_val.strip())
                        channels = line.split("\\n")
                        channels[0] = channels[0].replace("'", "")
                        if is_first_case:
                            n_channels = len(channels)
                            n_timepoints = len(channels[0].split(","))
                            is_first_case = False
                        elif len(channels) != n_channels:
                            raise ValueError(
                                f" Number of channels not equal in "
                                f"dataset, first case had {n_channels} channel "
                                f"but case number {n_cases+1} has "
                                f"{len(channels)}"
                            )
                        inst = np.zeros(shape=(n_channels, n_timepoints))
                        for c in range(len(channels)):
                            split = channels[c].split(",")
                            inst[c] = np.array([float(i) for i in split])
                    else:
                        line_parts = line.split(",")
                        if is_first_case:
                            is_first_case = False
                            n_timepoints = len(line_parts) - 1
                        class_val_list.append(line_parts[-1].strip())
                        split = line_parts[: len(line_parts) - 1]
                        inst = np.zeros(shape=(n_channels, n_timepoints))
                        inst[0] = np.array([float(i) for i in split])
                    instance_list.append(inst)
    return np.asarray(instance_list), np.asarray(class_val_list)


def load_from_tsv_file(full_file_path_and_name):
    """Load data from a .tsv file into a numpy array.

    tsv files are simply csv files with the class value as the first value. They only
    store equal length, univariate data, so are simple.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .tsv file to read.

    Returns
    -------
    data: np.ndarray
        time series data, np.ndarray (n_cases, 1, n_timepoints)
    y : np.ndarray of string or int
        target variable

    """
    df = pd.read_csv(full_file_path_and_name, sep="\t", header=None)
    y = df.pop(0).values
    df.columns -= 1
    X = df.to_numpy()
    X = np.expand_dims(X, axis=1)
    return X, y


def _convert_tsf_to_hierarchical(
    data: pd.DataFrame,
    metadata,
    freq: Optional[str] = None,
    value_column_name: str = "series_value",
) -> pd.DataFrame:
    """Convert the data from default_tsf to pd_multiindex_hier.

    Parameters
    ----------
    data : pd.DataFrame
        nested values dataframe
    metadata : Dict
        tsf file metadata
    freq : str, optional
        pandas compatible time frequency, by default None
        if not specified it's automatically mapped from the tsf frequency to a pandas
        frequency
    value_column_name: str, optional
        The name of the column that contains the values, by default "series_value"

    Returns
    -------
    pd.DataFrame
        hierarchical multiindex pd.Dataframe
    """
    df = data.copy()

    if freq is None:
        freq_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
            "yearly": "YS",
        }
        freq = freq_map[metadata["frequency"]]

    # create the time index
    if "start_timestamp" in df.columns:
        df["timestamp"] = df.apply(
            lambda x: pd.date_range(
                start=x["start_timestamp"], periods=len(x[value_column_name]), freq=freq
            ),
            axis=1,
        )
        drop_columns = ["start_timestamp"]
    else:
        df["timestamp"] = df.apply(
            lambda x: pd.RangeIndex(start=0, stop=len(x[value_column_name])), axis=1
        )
        drop_columns = []

    # pandas implementation of multiple column explode
    # can be removed and replaced by explode if we move to pandas version 1.3.0
    columns = [value_column_name, "timestamp"]
    index_columns = [c for c in list(df.columns) if c not in drop_columns + columns]
    result = pd.DataFrame({c: df[c].explode() for c in columns})
    df = df.drop(columns=columns + drop_columns).join(result)
    if df["timestamp"].dtype == "object":
        df = df.astype({"timestamp": "int64"})
    df = df.set_index(index_columns + ["timestamp"])
    df = df.astype({value_column_name: "float"}, errors="ignore")

    return df


def load_from_tsf_file(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
    return_type="tsf_default",
):
    """
    Convert the contents in a .tsf file into a dataframe.

    This code was extracted from
    https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py.

    Parameters
    ----------
    full_file_path_and_name : str
        The full path to the .tsf file.
    replace_missing_vals_with : str, default="NAN"
        A term to indicate the missing values in series in the returning dataframe.
    value_column_name : str, default="series_value"
        Any name that is preferred to have as the name of the column containing series
        values in the returning dataframe.
    return_type : str - "pd_multiindex_hier" or "tsf_default" (default)
        - "tsf_default" = container that faithfully mirrors tsf format from the original
            implementation in: https://github.com/rakshitha123/TSForecasting/
            blob/master/utils/data_loader.py.

    Returns
    -------
    loaded_data : pd.DataFrame
        The converted dataframe containing the time series.
    metadata : dict
        The metadata for the forecasting problem. The dictionary keys are:
        "frequency", "forecast_horizon", "contain_missing_values",
        "contain_equal_length"

    Raises
    ------
    URLError or HTTPError
        If the website is not accessible.
    ValueError
        If a dataset name that does not exist on the repo is given or if a
        webpage is requested that does not exist.
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

    with open(full_file_path_and_name, encoding="cp1252") as file:
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
                                if line_content[1].lower() == "true":
                                    contain_missing_values = True
                                else:
                                    contain_missing_values = False
                            elif line.startswith("@equallength"):
                                if line_content[1].lower() == "false":
                                    contain_equal_length = False
                                else:
                                    contain_equal_length = True
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


def load_forecasting(name, extract_path=None, return_metadata=False):
    """Download/load forecasting problem from https://forecastingdata.org/.

    Parameters
    ----------
    name : string, file name to load from
    extract_path : optional (default = None)
        Path of the location for the data file. If none, data is written to
        os.path.dirname(__file__)/data/
    return_metadata : boolean, default = False
        If True, returns a tuple (data, metadata)

    Returns
    -------
    X: pd.DataFrame
        Data stored in a dataframe, each column a series
    metadata: dict, optional
        returns the following metadata
        frequency,forecast_horizon,contain_missing_values,contain_equal_length

    Raises
    ------
    URLError or HTTPError
        If the website is not accessible.
    ValueError
        If a dataset name that does not exist on the repo is given or if a
        webpage is requested that does not exist.

    Examples
    --------
    >>> from aeon.datasets import load_forecasting
    >>> X=load_forecasting("m1_yearly_dataset") # doctest: +SKIP
    """
    # Allow user to have non standard extract path
    from aeon.datasets.tsf_datasets import tsf_all

    if extract_path is not None:
        local_module = extract_path
        local_dirname = ""
    else:
        local_module = MODULE
        local_dirname = "data"

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    # Check if data already in extract path or, if extract_path None,
    # in datasets/data directory
    if name not in get_downloaded_tsf_datasets(extract_path):
        # Dataset is not already present in the datasets directory provided.
        # If it is not there, download and install it.
        if name in tsf_all.keys():
            id = tsf_all[name]
            if extract_path is None:
                local_dirname = "local_data"
            if not os.path.exists(os.path.join(local_module, local_dirname)):
                os.makedirs(os.path.join(local_module, local_dirname))
        else:
            raise ValueError(
                f"File name {name} is not in the list of valid files to download"
            )
        if name not in get_downloaded_tsf_datasets(
            os.path.join(local_module, local_dirname)
        ):
            url = f"https://zenodo.org/record/{id}/files/{name}.zip"
            file_save = f"{local_module}/{local_dirname}/{name}.zip"
            if not os.path.exists(file_save):
                req = Request(url, method="HEAD")
                try:
                    # Perform the request
                    response = urlopen(req, timeout=60)
                    # Check the status code of the response, if 200 incorrect input args
                    if response.status != 200:
                        raise ValueError(
                            "The file does not exist on the server which "
                            "returned a File Not Found (200)"
                        )
                except Exception as e:
                    raise e
                try:
                    _download_and_extract(
                        url,
                        extract_path=extract_path,
                    )
                except zipfile.BadZipFile:
                    raise ValueError(
                        f"Invalid dataset name ={name} is  available on extract path ="
                        f"{extract_path} or https://zenodo.org/"
                        f" but it is not correctly formatted.",
                    )

            if not os.path.exists(
                f"{local_module}/{local_dirname}/{name}/" f"{name}.tsf"
            ):
                z = zipfile.ZipFile(file_save, "r")
                z.extractall(f"{local_module}/{local_dirname}/{name}/")
    full_name = f"{local_module}/{local_dirname}/{name}/{name}.tsf"
    data, meta = load_from_tsf_file(full_file_path_and_name=full_name)
    if return_metadata:
        return data, meta
    return data


def load_regression(
    name: str,
    split=None,
    extract_path=None,
    return_metadata: bool = False,
    load_equal_length: bool = True,
    load_no_missing: bool = True,
):
    """Download/load regression problem.

    Download from either https://timeseriesclassification.com or, if that fails,
    http://tseregression.org/.

    If you want to load a problem from a local file, specify the
    location in ``extract_path``. This function assumes the data is stored in format
    <extract_path>/<name>/<name>_TRAIN.ts and <extract_path>/<name>/<name>_TEST.ts.
    If you want to load a file directly from a full path, use the function
    `load_from_ts_file`` directly. If you do not specify ``extract_path``, or if the
    problem is not present in ``extract_path`` it will attempt to download the data
    from https://timeseriesclassification.com or, if that fails,
    http://tseregression.org/.

    The list of problems this function can download from the website is in
    ``datasets/tser_lists.py`` called ``tser_soton``. This function can load timestamped
    data, but it does not store the time stamps. The time stamp loading is fragile,
    it will only work if all data are floats.

    Data is assumed to be in the standard .ts format: each row is a (possibly
    multivariate) time series. Each dimension is separated by a colon, each value in
    a series is comma separated. For an example TSER problem see
    aeon.datasets.data.Covid3Month. Some of the original problems are unequal length
    and have missing values. By default, this function loads equal length no
    missing value versions of the files that have been used in experimental studies.
    These have suffixes `_eq` or `_nmv` after the name.
    If you want to load a different version, set the flags load_equal_length and/or
    load_no_missing to true. If present, the function will then load these versions
    if it can. aeon supports loading series with missing values and or unequal
    length between series, but it does not support loading multivariate series where
    lengths differ between channels. The original PGDALIA is in this format. The data
    PGDALIA_eq has length normalised series. If a problem has unequal length series
    and missing values, it is assumed to be of the form <name>_eq_nmv_TRAIN.ts and
    <name>_eq_nmv_TEST.ts. There are currently no problems in the archive with
    missing and unequal length.


    Parameters
    ----------
    name : string
        Name of the problem to load or download.
    extract_path : None or str, default = None
        Path of the location for the data file. If None, data is written to
        os.path.dirname(__file__)/local_data/<name>/.
    split : None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    return_metadata : boolean, default = False
        If True, returns a tuple (X, y, metadata)
    load_equal_length : boolean, default=True
        This is for the case when the standard release has unequal length series. The
        downloaded zip for these contain a version made equal length through
        truncation. These versions all have the suffix _eq after the name. If this
        flag is set to True, the function first attempts to load files called
        <name>_eq_TRAIN.ts/TEST.ts. If these are not present, it will load the normal
        version.
    load_no_missing : boolean, default=True
        This is for the case when the standard release has missing values. The
        downloaded zip for these contain a version with imputed missing values. These
        versions all have the suffix _nmv after the name. If this
        flag is set to True, the function first attempts to load files called
        <name>_nmv_TRAIN.ts/TEST.ts. If these are not present, it will load the normal
        version.

    Returns
    -------
    X: np.ndarray or list of np.ndarray
    y: np.ndarray
        The target response variable for each case in X
    metadata: dict, optional
        returns the following metadata
        'problemname',timestamps, missing,univariate,equallength.
        targetlabel should be true, and classlabel false

    Examples
    --------
    >>> from aeon.datasets import load_regression
    >>> X, y=load_regression("FloodModeling1") # doctest: +SKIP
    """
    if extract_path is not None:
        local_module = extract_path
        local_dirname = ""
    else:
        local_module = MODULE
        local_dirname = "data"
    error_str = (
        f"File name {name} is not in the list of valid files to download,"
        f"see aeon.datasets.tser_datasetss.tser_soton for the list. "
        f"If it is one tsc.com but not on the list, it means it may not "
        f"have been fully validated. Download it from the website."
    )
    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    path = os.path.join(local_module, local_dirname)
    if name not in get_downloaded_tsc_tsr_datasets(extract_path):
        if name in tser_soton:
            if extract_path is None:
                local_dirname = "local_data"
                if not os.path.exists(os.path.join(local_module, local_dirname)):
                    os.makedirs(os.path.join(local_module, local_dirname))
                path = os.path.join(local_module, local_dirname)
        else:
            raise ValueError(error_str)
        if name not in get_downloaded_tsc_tsr_datasets(
            os.path.join(local_module, local_dirname)
        ):
            # Check if on timeseriesclassification.com
            try_monash = False
            url = f"https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
            # Test if file exists
            req = Request(url, method="HEAD")
            try:
                # Perform the request
                response = urlopen(req, timeout=60)
                # Check the status code of the response
                if response.status != 200:
                    try_monash = True
            except (URLError, HTTPError):
                # If there is an HTTP it might mean the file does not exist
                try_monash = True
            else:
                try:
                    _download_and_extract(
                        url,
                        extract_path=extract_path,
                    )
                except zipfile.BadZipFile:
                    try_monash = True
            if try_monash:
                # Try on monash
                if name in tser_monash.keys():
                    id = tser_monash[name]
                    url_train = f"https://zenodo.org/record/{id}/files/{name}_TRAIN.ts"
                    url_test = f"https://zenodo.org/record/{id}/files/{name}_TEST.ts"
                    full_path = os.path.join(path, name)
                    if not os.path.exists(full_path):
                        os.makedirs(full_path)

                    train_save = f"{full_path}/{name}_TRAIN.ts"
                    test_save = f"{full_path}/{name}_TEST.ts"
                    try:
                        urlretrieve(url_train, train_save)
                        urlretrieve(url_test, test_save)
                    except Exception:
                        raise ValueError(error_str)
                else:
                    raise ValueError(error_str)
    # Test for non missing or equal length versions
    dir_name = name
    if load_equal_length:
        # If there exists a version with equal length, load that
        train = os.path.join(path, f"{name}/{name}_eq_TRAIN.ts")
        test = os.path.join(path, f"{name}/{name}_eq_TRAIN.ts")
        if os.path.exists(train) and os.path.exists(test):
            name = name + "_eq"
    if load_no_missing:
        train = os.path.join(path, f"{name}/{name}_nmv_TRAIN.ts")
        test = os.path.join(path, f"{name}/{name}_nmv_TRAIN.ts")
        if os.path.exists(train) and os.path.exists(test):
            name = name + "_nmv"

    X, y, meta = _load_saved_dataset(
        name=name,
        dir_name=dir_name,
        split=split,
        local_module=local_module,
        local_dirname=local_dirname,
        return_meta=True,
    )
    if return_metadata:
        return X, y, meta
    return X, y


def load_classification(
    name,
    split=None,
    extract_path=None,
    return_metadata=False,
    load_equal_length: bool = True,
    load_no_missing: bool = True,
):
    """Load a classification dataset.

    This function loads TSC problems into memory, downloading from
    https://timeseriesclassification.com/ if the data is not available at the
    specified local path. If you want to load a problem from a local file, specify the
    location in ``extract_path``. This function assumes the data is stored in format
    ``<extract_path>/<name>/<name>_TRAIN.ts`` and
    ``<extract_path>/<name>/<name>_TEST.ts.`` If you want to load a file directly
    from a full path, use the function `load_from_ts_file`` directly. If you do not
    specify ``extract_path``, it will set the path to ``aeon/datasets/local_data``. If
    the  problem is not present in ``extract_path`` it will attempt to download the data
    from https://timeseriesclassification.com/.

    This function can load timestamped data, but it does not store the time stamps.
    The time stamp loading is fragile, it will only work if all data are floats.

    Data is assumed to be in the standard .ts format: each row is a (possibly
    multivariate) time series. Each dimension is separated by a colon, each value in
    a series is comma separated. For examples see aeon.datasets.data. ArrowHead
    is an example of a univariate equal length problem, BasicMotions an equal length
    multivariate problem. See https://www.aeon-toolkit.org/en/stable/api_reference
    /file_specifications/ts.html for formatting details.

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in tsc_datasets is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from www.timeseriesclassification.com, saving it to
        the extract_path.
    split : None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    extract_path : str, default=None
        the path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/local_data/`. If a path is given, it can be absolute,
        e.g. C:/Temp/ or relative, e.g. Temp/ or ./Temp/.
    return_metadata : boolean, default = True
        If True, returns a tuple (X, y, metadata)
    load_equal_length : boolean, default=True
        This is for the case when the standard release has unequal length series. The
        downloaded zip for these contain a version made equal length through
        truncation. These versions all have the suffix _eq after the name. If this
        flag is set to True, the function first attempts to load files called
        <name>_eq_TRAIN.ts/TEST.ts. If these are not present, it will load the normal
        version.
    load_no_missing : boolean, default=True
        This is for the case when the standard release has missing values. The
        downloaded zip for these contain a version with imputed missing values. These
        versions all have the suffix _nmv after the name. If this
        flag is set to True, the function first attempts to load files called
        <name>_nmv_TRAIN.ts/TEST.ts. If these are not present, it will load the normal
        version.

    Returns
    -------
    X: np.ndarray or list of np.ndarray
    y: np.ndarray
        The class labels for each case in X
    metadata: dict, optional
        returns the following metadata
        'problemname',timestamps, missing,univariate,equallength, class_values
        targetlabel should be false, and classlabel true

    Raises
    ------
    URLError or HTTPError
        If the website is not accessible.
    ValueError
        If a dataset name that does not exist on the repo is given or if a
        webpage is requested that does not exist.

    Examples
    --------
    >>> from aeon.datasets import load_classification
    >>> X, y = load_classification(name="ArrowHead")  # doctest: +SKIP
    """
    if extract_path is not None:
        local_module = extract_path
        local_dirname = None
    else:
        local_module = MODULE
        local_dirname = "data"
    if local_dirname is None:
        path = local_module
    else:
        path = os.path.join(local_module, local_dirname)
    if not os.path.exists(path):
        os.makedirs(path)
    if name not in get_downloaded_tsc_tsr_datasets(path):
        if extract_path is None:
            local_dirname = "local_data"
            path = os.path.join(local_module, local_dirname)
        else:
            path = extract_path
        if not os.path.exists(path):
            os.makedirs(path)
        if name not in get_downloaded_tsc_tsr_datasets(path):
            # Check if on timeseriesclassification.com
            url = f"https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
            # Test if file exists to generate more informative error
            req = Request(url, method="HEAD")
            try_zenodo = False
            error_str = (
                f"Invalid dataset name ={name} that is not available on extract path "
                f"={extract_path}. Nor is it available on "
                f"https://timeseriesclassification.com/ or zenodo."
            )
            try:
                # Perform the request
                response = urlopen(req, timeout=60)
                # Check the status code of the response, if 200 incorrect input args
                if response.status != 200:
                    try_zenodo = True
            except (URLError, HTTPError):
                # If there is an HTTP it might mean the file does not exist
                try_zenodo = True
            else:
                try:
                    _download_and_extract(
                        url,
                        extract_path=extract_path,
                    )
                except zipfile.BadZipFile:
                    try_zenodo = True
            if try_zenodo:
                # Try on ZENODO
                if name in tsc_zenodo.keys():
                    id = tsc_zenodo[name]
                    url_train = f"https://zenodo.org/record/{id}/files/{name}_TRAIN.ts"
                    url_test = f"https://zenodo.org/record/{id}/files/{name}_TEST.ts"
                    full_path = os.path.join(path, name)
                    if not os.path.exists(full_path):
                        os.makedirs(full_path)
                    train_save = f"{full_path}/{name}_TRAIN.ts"
                    test_save = f"{full_path}/{name}_TEST.ts"
                    try:
                        urlretrieve(url_train, train_save)
                        urlretrieve(url_test, test_save)
                    except Exception:
                        raise ValueError(error_str)
                else:
                    raise ValueError(error_str)

    # Test for discrete version (first suffix _disc), always use that if it exists
    dir_name = name
    # If there exists a version with _discr, load that
    train = os.path.join(path, f"{name}/{name}_disc*TRAIN.ts")
    test = os.path.join(path, f"{name}/{name}_disc*TEST.ts")
    train_match = glob.glob(train)
    test_match = glob.glob(test)
    if train_match and test_match:
        name = name + "_disc"
    if load_equal_length:
        # If there exists a version with equal length, load that
        train = os.path.join(path, f"{dir_name}", f"{name}_eq_TRAIN.ts")
        test = os.path.join(path, f"{dir_name}", f"{name}_eq_TEST.ts")
        if os.path.exists(train) and os.path.exists(test):
            name = name + "_eq"
    if load_no_missing:
        train = os.path.join(path, f"{dir_name}", f"{name}_nmv_TRAIN.ts")
        test = os.path.join(path, f"{dir_name}", f"{name}_nmv_TEST.ts")
        if os.path.exists(train) and os.path.exists(test):
            name = name + "_nmv"

    X, y, meta = _load_saved_dataset(
        name=name,
        dir_name=dir_name,
        split=split,
        local_module=local_module,
        local_dirname=local_dirname,
        return_meta=True,
    )
    # Check this is a classification problem
    if "classlabel" not in meta or not meta["classlabel"]:
        raise ValueError(
            f"You have tried to load a regression problem called {name} with "
            f"load_classifier. This will cause unintended consequences for any "
            f"classifier you build. If you want to load a regression problem, "
            f"use load_regression "
        )
    if return_metadata:
        return X, y, meta
    return X, y


def download_all_regression(extract_path=None):
    """Download and unpack all of the Monash TSER datasets.

    Parameters
    ----------
    extract_path: str or None, default = None
        where to download the fip file. If none, it goes in

    Raises
    ------
    URLError or HTTPError if the website is not accessible.
    """
    if extract_path is not None:
        local_module = extract_path
        local_dirname = ""
    else:
        local_module = MODULE
        local_dirname = "data"

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    url = (
        "https://zenodo.org/record/4632512/files/Monash_UEA_UCR_Regression_Archive.zip"
    )
    if extract_path is None:
        local_dirname = "local_data"
    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))

    file_save = f"{local_module}/{local_dirname}/Monash_UEA_UCR_Regression_Archive.zip"
    # Check if it already exists at this location, to avoid repeated download
    if not os.path.exists(file_save):
        try:
            urllib.request.urlretrieve(url, file_save)
        except Exception:
            raise ValueError(
                f"Unable to download {file_save} from {url}",
            )
    zipfile.ZipFile(file_save, "r").extractall(f"{local_module}/{local_dirname}/")


PROBLEM_TYPES = [
    "AUDIO",
    "DEVICE",
    "ECG",
    "EEG",
    "EMG",
    "EOG",
    "EPG",
    "FINANCIAL",
    "HAR",
    "HEMODYNAMICS",
    "IMAGE",
    "MEG",
    "MOTION",
    "OTHER",
    "SENSOR",
    "SIMULATED",
    "SPECTRO",
]


def get_dataset_meta_data(
    data_names=None,
    features=None,
    url="https://timeseriesclassification.com/aeon-toolkit/metadata.csv",
):
    """Retrieve dataset meta data from timeseriesclassification.com.

    Metadata includes the following information for each dataset:
    - Dataset: name of the problem, set the lists in tsc_datasets for valid names.
    - TrainSize: number of series in the default train set.
    - TestSize:	number of series in the default train set.
    - Length: length of the series. If the series are not all the same length,
        this is set to 0.
    - NumberClasses: number of classes in the problem.
    - Type: nature of the problem, one of PROBLEM_TYPES
    - Channels: number of channels. If univariate, this is 1.


    Parameters
    ----------
    data_names : list, default=None
        List of dataset names to retrieve meta data for. If None, all datasets are
        retrieved.
    features : String or List, default=None
        List of features to retrieve meta data for. Should be a subset of features
        listed above. Dataset field is always returned.
    url : String
        default = "https://timeseriesclassification.com/aeon-toolkit/metadata.csv"
        Location of the csv metadata file.

    Returns
    -------
     Pandas dataframe containing meta data for each dataset.

    Raises
    ------
    URLError or HTTPError if the website is not accessible.
    """
    # Check string is either a valid local path or responding web page
    if not os.path.isfile(url):
        parsed_url = urlparse(url)
        if not (bool(parsed_url.scheme) and bool(parsed_url.netloc)):
            raise ValueError(f"Invalid URL or file path {url}")

    if isinstance(features, str):
        features = [features]

    try:
        if features is None:
            df = pd.read_csv(url)
        else:
            features.append("Dataset")
            df = pd.read_csv(url, usecols=features)
        if data_names is not None:
            df = df[df["Dataset"].isin(data_names)]
    except Exception as e:
        raise e
    return df
