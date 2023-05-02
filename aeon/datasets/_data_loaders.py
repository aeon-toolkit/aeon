# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from aeon.datasets._data_dataframe_loaders import MODULE
from aeon.datasets.tsc_dataset_names import _list_available_datasets
from aeon.datatypes import convert

VALID_RETURN_TYPES = ["numpy3d", "numpy2d", "np3d", "np2d", "np_list", "nested_univ"]
DIRNAME = "data"
# MODULE = os.path.dirname(__file__)


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
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise IOError("invalid class label value")
                    if token_len == 2:
                        raise IOError(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
    return meta_data


def _load_data(file, meta_data):
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

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    series_length = 0
    y_values = []
    for line in file:
        line = line.strip().lower()
        line = line.replace("?", "NaN")
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
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
    replace_missing_vals_with : string, default="NaN"
    return_meta_data : boolean, default=True

    Returns
    -------
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, series_length) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, series_length)
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
    if return_meta_data:
        return data, y, meta_data
    return data, y


def _load_provided_dataset(
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
    Raise ValueException if the requested return type is not supported

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
    else:
        loaded_type = "numpy3D"

    if return_type == "nested_univ":
        X = convert(X, from_type="np-list", to_type="nested_univ")
        loaded_type = "nested_univ"
    elif meta_data["equallength"]:
        if return_type == "numpyflat" and X.shape[1] == 1:
            X = X.squeeze()
    if return_type is not None and loaded_type != return_type:
        X = convert(X, from_type=loaded_type, to_type=return_type)
    if return_X_y:
        return X, y
    else:  # TODO: do this better, do we want it all in dataframes?
        X = convert(X, from_type=loaded_type, to_type="nested_univ")
        X["class_val"] = pd.Series(y)
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


def _load_dataset(name, split, return_X_y=True, return_type=None, extract_path=None):
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
        "np-list": for unequal length series that cannot be storeed in numpy arrays
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
        local_dirname = extract_path
    else:
        local_module = MODULE
        local_dirname = "data"

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    if name not in _list_available_datasets(extract_path):
        if extract_path is None:
            local_dirname = "local_data"
        if not os.path.exists(os.path.join(local_module, local_dirname)):
            os.makedirs(os.path.join(local_module, local_dirname))
        if name not in _list_available_datasets(
            os.path.join(local_module, local_dirname)
        ):
            # Dataset is not already present in the datasets directory provided.
            # If it is not there, download and install it.
            url = "https://timeseriesclassification.com/Downloads/%s.zip" % name
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

    return _load_provided_dataset(
        name, split, return_X_y, return_type, local_module, local_dirname
    )
