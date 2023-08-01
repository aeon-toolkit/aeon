# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import urllib
import zipfile
from datetime import datetime
from distutils.util import strtobool
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from aeon.datasets._dataframe_loaders import DIRNAME, MODULE
from aeon.datasets.dataset_collections import (
    list_downloaded_tsc_tsr_datasets,
    list_downloaded_tsf_datasets,
)
from aeon.datatypes import MTYPE_LIST_HIERARCHICAL, convert

__all__ = [  # Load functions
    "load_from_tsfile",
    "load_from_tsf_file",
    "load_from_arff_file",
    "load_from_tsv_file",
    "load_classification",
    "load_forecasting",
    "load_regression",
    "download_all_regression",
]


# Return appropriate return_type in case an alias was used
def _alias_datatype_check(return_type):
    if return_type in ["numpy2d", "numpy2D", "np2d", "np2D"]:
        return_type = "numpyflat"
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
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise IOError("data tag should not have an associated value")
                return meta_data
            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise IOError(f"{tokens[0]} tag requires a boolean value")
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
                            raise IOError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise IOError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data


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
    series_length = 0
    y_values = []
    for line in file:
        line = line.strip().lower()
        line = line.replace("?", replace_missing_vals_with)
        channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            current_channels -= 1
        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels = current_channels
            if meta_data["equallength"]:
                series_length = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise IOError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"]:
                if current_channels > 1:
                    raise IOError(
                        f"Seen {current_channels} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
        if meta_data["equallength"]:
            current_length = series_length
        else:
            current_length = len(channels[0].split(","))
        np_case = np.zeros(shape=(n_channels, current_length))
        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length:
                raise IOError(
                    f"Unequal length series, in case {n_cases} meta "
                    f"data specifies all equal {series_length} but saw "
                    f"{len(single_channel)}"
                )
            np_case[i] = np.array(data_series)
        data.append(np_case)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            y_values.append(channels[n_channels])
    if meta_data["equallength"]:
        data = np.array(data)
    return data, np.asarray(y_values), meta_data


def load_from_tsfile(
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
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, series_length) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : target variable, np.ndarray of string or int
    meta_data : dict (optional).
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # Check file ends in .ts, if not, insert
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    # Open file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
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
    return_X_y=True,
    return_type=None,
    local_module=MODULE,
    local_dirname=DIRNAME,
    return_meta=False,
):
    """Load baked in time series classification datasets (helper function).

    Loads data from the provided files from aeon/datasets/data only.

    Parameters
    ----------
    name : string, file name to load from
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.
    return_data_type : str, optional, default = None
        "numpy3D"/"numpy3d"/"np3D": recommended for equal length series
        "numpy2D"/"numpy2d"/"np2d": can be used for univariate equal length series,
        although we recommend numpy3d, because some transformers do not work with
        numpy2d. If None will load 3D numpy or list of numpy
        There other options, see datatypes.SCITYPE_REGISTER, but these
        will not necessarily be supported longterm.
    local_module: default = os.path.dirname(__file__),
    local_dirname: default = "data"

    Raises
    ------
    Raise ValueError if the requested return type is not supported

    Returns
    -------
    X: Data stored in specified `return_type`
        The time series data for the problem, with n instances
    y: 1D numpy array of length n, only returned if return_X_y if True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.
    """
    if isinstance(split, str):
        split = split.upper()
    # This is also called in load_from_tsfile, but we need the value here and it
    # is required in load_from_tsfile since it is public
    return_type = _alias_datatype_check(return_type)

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        abspath = os.path.join(local_module, local_dirname, name, fname)
        X, y, meta_data = load_from_tsfile(abspath, return_meta_data=True)
    # if split is None, load both train and test set
    elif split is None:
        fname = name + "_TRAIN.ts"
        abspath = os.path.join(local_module, local_dirname, name, fname)
        X_train, y_train, meta_data = load_from_tsfile(abspath, return_meta_data=True)

        fname = name + "_TEST.ts"
        abspath = os.path.join(local_module, local_dirname, name, fname)
        X_test, y_test, meta_data_test = load_from_tsfile(
            abspath, return_meta_data=True
        )
        if meta_data["equallength"]:
            X = np.concatenate([X_train, X_test])
        else:
            X = X_train + X_test
        y = np.concatenate([y_train, y_test])
    else:
        raise ValueError("Invalid `split` value =", split)

    # All this is to allow for the user to configure to load into different data
    # structures. Its all for backward compatibility.
    if isinstance(X, list):
        loaded_type = "np-list"
    elif isinstance(X, np.ndarray):
        loaded_type = "numpy3D"

    if return_type == "nested_univ":
        X = convert(X, from_type="np-list", to_type="nested_univ")
        loaded_type = "nested_univ"
    elif meta_data["equallength"]:
        if return_type == "numpyflat" and X.shape[1] == 1:
            X = X.squeeze()
            loaded_type = "numpyflat"
    elif return_type is not None and loaded_type != return_type:
        X = convert(X, from_type=loaded_type, to_type=return_type)

    if return_X_y:
        if return_meta:
            return X, y, meta_data
        else:
            return X, y
    else:  # TODO: do this better, do we want it all in dataframes?
        X = convert(X, from_type=loaded_type, to_type="nested_univ")
        X["class_val"] = pd.Series(y)
        if return_meta:
            return X, meta_data
        else:
            return X


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
    urlretrieve(url, zip_file_name)

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
    name, split, return_X_y=True, return_type=None, extract_path=None, return_meta=False
):
    """Load time series classification datasets (helper function).

    Parameters
    ----------
    name : string, file name to load from
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.
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

    Raises
    ------
    Raise ValueException if the requested return type is not supported

    Returns
    -------
    X: Data stored in specified `return_type`
        The time series data for the problem, with n instances
    y: 1D numpy array of length n, only returned if return_X_y if True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.
    """
    # Allow user to have non standard extract path
    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = ""
    else:
        local_module = MODULE
        local_dirname = "data"

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    if name not in list_downloaded_tsc_tsr_datasets(extract_path):
        if extract_path is None:
            local_dirname = "local_data"
        if not os.path.exists(os.path.join(local_module, local_dirname)):
            os.makedirs(os.path.join(local_module, local_dirname))
        if name not in list_downloaded_tsc_tsr_datasets(
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
                    f"{extract_path}. Nor is it available on "
                    f"https://timeseriesclassification.com/.",
                ) from e

    return _load_saved_dataset(
        name,
        split=split,
        return_X_y=return_X_y,
        return_type=return_type,
        local_module=local_module,
        local_dirname=local_dirname,
        return_meta=return_meta,
    )


def load_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
    return_type="pd_multiindex_hier",
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
    return_type : str - "pd_multiindex_hier" (default), "tsf_default", or valid aeon
        mtype string for in-memory data container format specification of the
        return type:
        - "pd_multiindex_hier" = pd.DataFrame of aeon type `pd_multiindex_hier`
        - "tsf_default" = container that faithfully mirrors tsf format from the original
            implementation in: https://github.com/rakshitha123/TSForecasting/
            blob/master/utils/data_loader.py.
        - other valid mtype strings are Panel or Hierarchical mtypes in
            datatypes.MTYPE_REGISTER. If Panel or Hierarchical mtype str is given, a
            conversion to that mtype will be attempted

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

        if return_type != "default_tsf":
            loaded_data = _convert_tsf_to_hierarchical(
                loaded_data, metadata, value_column_name=value_column_name
            )
            if (
                loaded_data.index.nlevels == 2
                and return_type not in MTYPE_LIST_HIERARCHICAL
            ):
                loaded_data = convert(
                    loaded_data, from_type="pd-multiindex", to_type=return_type
                )
            else:
                loaded_data = convert(
                    loaded_data, from_type="pd_multiindex_hier", to_type=return_type
                )

        return loaded_data, metadata


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
        time series data, np.ndarray (n_cases, n_channels, series_length)
    y : target variable, np.ndarray of string or int
    """
    instance_list = []
    class_val_list = []
    data_started = False
    is_multi_variate = False
    is_first_case = True
    n_cases = 0
    with open(full_file_path_and_name, "r", encoding="utf-8") as f:
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
                n_channels = 1
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
                                f"dataset, first case had {n_channels} "
                                f"but {n_cases+1} case hase "
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
        time series data, np.ndarray (n_cases, 1, series_length)
    y : target variable, np.ndarray of string or int

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
    freq: str = None,
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
        aeon pd_multiindex_hier mtype
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
):
    """
    Convert the contents in a .tsf file into a dataframe.

    This code was extracted from
    https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py.

    Parameters
    ----------
    full_file_path_and_name: str
        The full path to the .tsf file.
    replace_missing_vals_with: str, default="NAN"
        A term to indicate the missing values in series in the returning dataframe.
    value_column_name: str, default="series_value"
        Any name that is preferred to have as the name of the column containing series
        values in the returning dataframe.

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
        return loaded_data, metadata


def load_forecasting(name, extract_path=None, return_metadata=True):
    """Download/load forecasting problem from https://forecastingdata.org/.

    Parameters
    ----------
    name : string, file name to load from
    extract_path : optional (default = None)
        Path of the location for the data file. If none, data is written to
        os.path.dirname(__file__)/data/
    return_metadata : boolean, default = True
        If True, returns a tuple (data, metadata)

    Raises
    ------
    Raise ValueException if the requested return type is not supported

    Returns
    -------
    X: Data stored in a dataframe, each column a series
    metadata: optional
        returns the following meta data
        frequency,forecast_horizon,contain_missing_values,contain_equal_length

    Example
    -------
    >>> from aeon.datasets import load_forecasting
    >>> X, meta=load_forecasting("m1_yearly_dataset") #DOCTEST +skip
    """
    # Allow user to have non standard extract path
    from aeon.datasets.tsf_data_lists import tsf_all

    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = ""
    else:
        local_module = MODULE
        local_dirname = "data"

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    # Check if data already in extract path or, if extract_path None,
    # in datasets/data directory
    if name not in list_downloaded_tsf_datasets(extract_path):
        if extract_path is None:
            local_dirname = "local_data"
        if not os.path.exists(os.path.join(local_module, local_dirname)):
            os.makedirs(os.path.join(local_module, local_dirname))
        if name not in list_downloaded_tsc_tsr_datasets(
            os.path.join(local_module, local_dirname)
        ):
            # Dataset is not already present in the datasets directory provided.
            # If it is not there, download and install it.
            if name in tsf_all.keys():
                id = tsf_all[name]
            else:
                raise ValueError(
                    f"File name {name} is not in the list of valid files to download"
                )
            url = f"https://zenodo.org/record/{id}/files/{name}.zip"
            file_save = f"{local_module}/{local_dirname}/{name}.zip"
            if not os.path.exists(file_save):
                try:
                    urllib.request.urlretrieve(url, file_save)
                except Exception:
                    raise ValueError(
                        f"Invalid dataset name ={name} is not available on extract path"
                        f" {extract_path}.\n Nor is it available on "
                        f"https://forecastingdata.org/ via path {url}",
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


def load_regression(name, split=None, extract_path=None, return_metadata=True):
    """Download/load forecasting problem from https://forecastingdata.org/.

    Parameters
    ----------
    name : string, file name to load from
    extract_path : optional (default = None)
        Path of the location for the data file. If none, data is written to
        os.path.dirname(__file__)/data/<name>/
    split : None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    return_metadata : boolean, default = True
        If True, returns a tuple (X, y, metadata)

    Raises
    ------
    Raise ValueException if the requested return type is not supported

    Returns
    -------
    X: np.ndarray or list of np.ndarray
    y: numpy array
        The target response variable for each case in X
    metadata: optional
        returns the following meta data
        'problemname',timestamps, missing,univariate,equallength.
        targetlabel should be true, and classlabel false

    Example
    -------
    >>> from aeon.datasets import load_regression
    >>> X, y, meta=load_regression("FloodModeling1") #DOCTEST +Skip
    """
    from aeon.datasets.tser_data_lists import tser_all

    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = ""
    else:
        local_module = MODULE
        local_dirname = "data"

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    if name not in list_downloaded_tsc_tsr_datasets(extract_path):
        if extract_path is None:
            local_dirname = "local_data"
        if not os.path.exists(os.path.join(local_module, local_dirname)):
            os.makedirs(os.path.join(local_module, local_dirname))
        if name not in list_downloaded_tsc_tsr_datasets(
            os.path.join(local_module, local_dirname)
        ):
            if name in tser_all.keys():
                id = tser_all[name]
            else:
                raise ValueError(
                    f"File name {name} is not in the list of valid files to download"
                )
            # Dataset is not already present in the datasets directory provided.
            # If it is not there, download and install it.
            url_train = f"https://zenodo.org/record/{id}/files/{name}_TRAIN.ts"
            url_test = f"https://zenodo.org/record/{id}/files/{name}_TEST.ts"
            if not os.path.exists(f"{local_module}/{local_dirname}/{name}"):
                os.makedirs(f"{local_module}/{local_dirname}/{name}")

            train_save = f"{local_module}/{local_dirname}/{name}/{name}_TRAIN.ts"
            test_save = f"{local_module}/{local_dirname}/{name}/{name}_TEST.ts"
            try:
                urllib.request.urlretrieve(url_train, train_save)
                urllib.request.urlretrieve(url_test, test_save)
            except Exception:
                raise ValueError(
                    f"Invalid dataset name ={name} one or both of TRAIN and TEST is "
                    f"not available on path ={local_module}/{local_dirname}/{name}.\n "
                    f"Nor is it available on tseregression.org via path {url_train} "
                    f"or {url_test}"
                )
    #            zipfile.ZipFile(file_save, "r").extractall(f"{extract_path}/{name}/")
    return _load_saved_dataset(
        name=name,
        split=split,
        local_module=local_module,
        local_dirname=local_dirname,
        return_meta=return_metadata,
    )


def load_classification(name, split=None, extract_path=None, return_metadata=True):
    """Load a classification dataset.

    Loads a TSC dataset from extract_path, or from timeseriesclassification.com,
    if not on extract path.

    Data is assumed to be in the standard .ts format: each row is a (possibly
    multivariate) time series.
    Each dimension is separated by a colon, each value in a series is comma
    separated. For examples see aeon.datasets.data.tsc. ArrowHead is an example of
    a univariate equal length problem, BasicMotions an equal length multivariate
    problem.

    Data is stored in extract_path/name/name.ts, extract_path/name/name_TRAIN.ts and
    extract_path/name/name_TEST.ts.

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in tsc_data_lists is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from www.timeseriesclassification.com, saving it to
        the extract_path.
    split : None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    extract_path : str, default=None
        the path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/data/`. If a path is given, it can be absolute,
        e.g. C:/Temp/ or relative, e.g. Temp/ or ./Temp/.
    return_metadata : boolean, default = True
        If True, returns a tuple (X, y, metadata)

    Returns
    -------
    X: np.ndarray or list of np.ndarray
    y: numpy array
        The class labels for each case in X
    metadata: optional
        returns the following meta data
        'problemname',timestamps, missing,univariate,equallength, class_values
        targetlabel should be false, and classlabel true

    Examples
    --------
    >>> from aeon.datasets import load_classification
    >>> X, y, meta = load_classification(name="ArrowHead") #DOCTEST +Skip
    """
    return _load_tsc_dataset(
        name,
        split,
        return_X_y=True,
        extract_path=extract_path,
        return_meta=return_metadata,
    )


def download_all_regression(extract_path=None):
    """Download and unpack all of the Monash TSER datasets.

    Arguments
    ---------
    extract_path: str or None, default = None
        where to download the fip file. If none, it goes in
    """
    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
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
