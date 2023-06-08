# -*- coding: utf-8 -*-
"""
List of datasets available for classification, regression and forecasting archives.

The data can also be used for clustering.

Classification data can be downloaded directly from the timeseriesclassification.com
archive.

Regression data is and forecasting


Classification/Regression
-------------------------
Classification data available are listed in tsc_data_lists.py. Regression problems
are in tser_data_lists.py. The data can be downloaded and loaded with
>>> from aeon.datasets import load_from_tsfile

If the data is already stored on disk, you can just load it directly
Metadata is loaded from the file header, and return is optional

into 3D numpy arrays (n_cases, n_channels, n_timepoints) if equal length

Forecasting
-----------


"""

__author__ = ["Tony Bagnall"]
__all__ = [
    "list_downloaded_tsc_tsr_datasets",
    "list_downloaded_tsf_datasets",
    "list_available_tser_datasets",
    "list_available_tsf_datasets",
]
import os

from aeon.datasets._dataframe_loaders import MODULE
from aeon.datasets.tsc_data_lists import multivariate, univariate
from aeon.datasets.tser_data_lists import tser_all
from aeon.datasets.tsf_data_lists import tsf_all


def list_available_tser_datasets(name=None):
    """List available tser data."""
    if name is None:  # List them all
        return sorted(list(tser_all))
    if name in tser_all:
        return True
    return False


def list_available_tsf_datasets(name=None):
    """List available tsf data."""
    if name is None:  # List them all
        return sorted(list(tsf_all))
    if name in tsf_all:
        return True
    return False


def list_available_tsc_datasets(name=None):
    """List available tsf data.

    Parameters
    ----------
    name: str or None, default = None
        query dataset name

    Returns
    -------
    if name is None
        return combined list of multivariate and univaraite
    else
        return True if name is in either multivariate or univaraite
    """
    if name is None:  # List them all
        merged_set = univariate.union(multivariate)
        return sorted(list(merged_set))
    if name in univariate or name in multivariate:
        return True
    return False


def list_downloaded_tsc_tsr_datasets(extract_path=None):
    """Return a list of all the currently downloaded datasets.

    To count as available, each directory in extract_path <dir_name> in the
    extract_path must contain  files called <dir_name>_TRAIN.ts and <dir_name>_TEST.ts.

    Parameters
    ----------
    extract_path: string, default None
        root directory where to look for files, if None defaults to aeon/datasets/data

    Returns
    -------
    datasets : List
        List of the names of datasets downloaded

    """
    if extract_path is None:
        data_dir = os.path.join(MODULE, "data")
    else:
        data_dir = extract_path
    datasets = []
    for name in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, name)
        if os.path.isdir(sub_dir):
            all_files = os.listdir(sub_dir)
            if name + "_TRAIN.ts" in all_files and name + "_TEST.ts" in all_files:
                datasets.append(name)
    return datasets


def list_downloaded_tsf_datasets(extract_path=None):
    """Return a list of all the currently downloaded datasets.

    To count as available, each directory in extract_path <dir_name> in the
    extract_path must contain  files called <dir_name>_TRAIN.ts and <dir_name>_TEST.ts.

    Parameters
    ----------
    extract_path: string, default None
        root directory where to look for files, if None defaults to aeon/datasets/data

    Returns
    -------
    datasets : List
        List of the names of datasets downloaded

    """
    if extract_path is None:
        data_dir = os.path.join(MODULE, "data")
    else:
        data_dir = extract_path
    datasets = []
    for name in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, name)
        if os.path.isdir(sub_dir):
            all_files = os.listdir(sub_dir)
            if name + ".tsf" in all_files:
                datasets.append(name)
    return datasets
