"""
Archives for time series classification, regression and forecasting.

The classification and regression data can also be used for clustering.
Classification data can be downloaded directly from the zenodo
archive https://zenodo.org/communities/tsml or in code. Whole archives can be
downloaded with download_archive, single problems with load_classification,
load_regression and load_foreasting.

# Whole archives downloadable to TSML Zenodo from https://zenodo.org/records/<ID> ,
with IDs listed in tsml_archives

"""

__maintainer__ = ["TonyBagnall"]

__all__ = [
    "get_downloaded_tsc_tsr_datasets",
    "get_downloaded_tsf_datasets",
    "get_available_tser_datasets",
    "get_available_tsf_datasets",
    "tsml_archives",
    "tsml_zip_names",
]

import os

import aeon
from aeon.datasets.tsc_datasets import multivariate, univariate
from aeon.datasets.tser_datasets import tser_monash, tser_soton
from aeon.datasets.tsf_datasets import tsf_all

MODULE = os.path.join(os.path.dirname(aeon.__file__), "datasets")

tsml_archives = {
    "Synthetic Unequal Length UCR Time Series Classification Datasets 2026": 18300287,
    "TSML Extended Time Series Extrinsic Regression Archive 2024": 11236865,
    "TSML Multivariate Time Series Classification Archive 2018": 11206331,
    "Time Series Classification Bakeoff Redux Datasets 2024": 11206358,
    "UCR Time Series Classification Archive 2018": 11198697,
    "TSML Imbalanced Univariate Time Series Classification Archive 2025": 18641021,
    "UCR": 11206331,
    "UEA": 11206331,
    "TSR": 11236865,
    "Unequal": 18300287,
    "Imbalanced": 18641021,
}
tsml_zip_names = {
    "UCR": "UCR%20Archive%202018.zip",
    "UEA": "TSML MV Archive 2018.zip",
    "TSR": "TSER%20Archive%20Datasets%202024.zip",
    "Unequal": "Unequal Length UCR Datasets 2026.zip",
    "Imbalanced": "UCR_Imbalanced_9_1.zip",
}


def get_available_tser_datasets(name="tser_soton", return_list=True):
    """List available tser data as specified by lists.

    Parameters
    ----------
    name : str or None, default = "tser_soton"
        One of the names in tser_soton or tser_monash, or None to list all.

    return_list : bool, default = True
        Whether to return problems as a list or a set.

    Returns
    -------
        list
            List of available datasets.
    """
    if name == "tser_soton":  # List them all
        if return_list:
            return sorted(list(set(tser_soton).union(set(tser_monash))))
        else:
            return set(tser_soton)
    if name == "tser_monash":
        if return_list:
            return sorted(list(tser_monash))
        else:
            return tser_monash
    return name in tser_soton


def get_available_tsf_datasets(name=None):
    """List available tsf data."""
    if name is None:  # List them all
        return sorted(list(tsf_all))
    return name in tsf_all


def get_available_tsc_datasets(name=None):
    """List available local TSC data.

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
        merged_set = set(univariate).union(set(multivariate))
        return sorted(list(merged_set))
    return name in univariate or name in multivariate


def get_downloaded_tsc_tsr_datasets(extract_path=None):
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


def get_downloaded_tsf_datasets(extract_path=None):
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
