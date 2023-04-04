# -*- coding: utf-8 -*-
import numpy as np

from aeon.datasets._data_io import load_from_tsfile_to_dataframe

VALID_RETURN_TYPES = ["numpy3d", "numpy2d", "np3d", "np2d", "numpy_list", "nested_univ"]


def _load_header_info(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : input file

    Returns
    -------
    meta_data : dictionary with the data characteristics stored in the header.
    """
    meta_data = {
        "problemname": "none",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equalLength": True,
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
            if key in meta_data.keys():
                if key == "data":
                    if line != "@data":
                        raise IOError("data tag should not have an associated value")
                    return meta_data
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


def _load_data(file, meta_data):
    """Load data from a file  with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : input file
    meta_data : hash table
        with meta data in the file header

    Returns
    -------
    list of numpy arrays of floats: the time series
    numpy array of strings: the class/target variable values
    meta_data enhanced with number of channels and series length

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    series_length = 0
    y_values = []
    for line in file:
        line = line.strip().lower()
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
    return data, np.asarray(y_values), meta_data


def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=True,
):
    """Load time series .ts file into X and (optionally) y.

    Raises
    ------
    IOError if the load fails
    """
    # Open file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        # if equal load to 3D numpy
        if meta_data["equallength"]:
            data, y, meta_data = _load_data(file, meta_data)
        # otherwise return dataframe for now, soon to be list of numpy
        else:
            data, y = load_from_tsfile_to_dataframe(
                full_file_path_and_name=full_file_path_and_name,
                return_separate_X_and_y=True,
                replace_missing_vals_with=replace_missing_vals_with,
            )
    if return_meta_data:
        return data, y, meta_data
    return data, y
