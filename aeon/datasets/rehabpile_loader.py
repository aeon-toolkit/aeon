# aeon/datasets/rehabpile_loader.py

import os
import re
from pathlib import Path
from typing import List, Literal, Tuple, Union
from urllib.error import HTTPError
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd

# The root URL for the RehabPile dataset collection
REHABPILE_ROOT_URL = "https://maxime-devanne.com/datasets/RehabPile/"

# The list of base collections as defined in your script
REHABPILE_COLLECTIONS = [
    "EHE_reg",
    "IRDS_clf_bn",
    "KERAAL_clf_bn",
    "KERAAL_clf_mc",
    "KIMORE_clf_bn",
    "KIMORE_reg",
    "KINECAL_clf_bn",
    "SPHERE_clf_bn",
    "UCDHE_clf_bn",
    "UCDHE_clf_mc",
    "UIPRMD_clf_bn",
    "UIPRMD_reg",
]


def _fetch_rehabpile_dataset_names() -> Tuple[List[str], List[str]]:
    """
    Scrape the RehabPile website to discover all available datasets.

    This function mimics the discovery logic from the provided bash script.
    It connects to the website, fetches the subfolder for each collection,
    and constructs the full dataset names.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing two lists: the first for classification dataset
        names and the second for regression dataset names.
    """
    classification_datasets = []
    regression_datasets = []

    for collection in REHABPILE_COLLECTIONS:
        collection_url = f"{REHABPILE_ROOT_URL}{collection}/"
        try:
            with urlopen(collection_url, timeout=30) as response:
                html_content = response.read().decode("utf-8")
        except HTTPError as e:
            print(
                f"Warning: Could not access {collection_url}. "
                f"Skipping. Error: {e}"
            )
            continue

        # Use regex to find all links to subdirectories, e.g., 'href="Subject_1/"'
        subfolders = re.findall(r'href="([^"/]+)/"', html_content)

        for folder in subfolders:
            dataset_name = f"{collection}_{folder}"
            if "clf" in collection:
                classification_datasets.append(dataset_name)
            elif "reg" in collection:
                regression_datasets.append(dataset_name)

    return sorted(classification_datasets), sorted(regression_datasets)


# Discover the datasets once when the module is imported
(
    _rehabpile_classification_datasets,
    _rehabpile_regression_datasets,
) = _fetch_rehabpile_dataset_names()


def rehabpile_classification_datasets() -> List[str]:
    """
    Return a list of all classification datasets in the RehabPile archive.

    Returns
    -------
    list[str]
        A list of dataset names for the classification task.
    """
    return _rehabpile_classification_datasets


def rehabpile_regression_datasets() -> List[str]:
    """
    Return a list of all regression datasets in the RehabPile archive.

    Returns
    -------
    list[str]
        A list of dataset names for the regression task.
    """
    return _rehabpile_regression_datasets


def load_rehabpile(
    name: str,
    split: Literal["train", "test"] = "test",
    extract_path: Union[str, Path, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from the RehabPile collection.

    This function downloads the required .npy files for a given RehabPile
    dataset if they are not already present in the extract_path. It then
    loads and returns the requested data split.

    The data is expected to be in pre-split numpy format (X_train.npy,
    y_train.npy, X_test.npy, y_test.npy).

    Parameters
    ----------
    name : str
        The name of the RehabPile dataset, e.g., "KIMORE_clf_bn_Subject_1".
        A full list can be obtained from `rehabpile_classification_datasets()`
        or `rehabpile_regression_datasets()`.
    split : {"train", "test"}, default="test"
        The split of the data to return.
    extract_path : str or Path, default=None
        The path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/local_data/`. The data will be stored in a
        subdirectory corresponding to its task (classification/ or regression/).

    Returns
    -------
    X : np.ndarray
        The time series data.
    y : np.ndarray
        The target labels or values.

    Raises
    ------
    ValueError
        If the dataset name is not found in the RehabPile collection or if the
        split is invalid.
    HTTPError
        If the dataset files cannot be downloaded from the server.
    """
    if name not in _rehabpile_classification_datasets and name not in _rehabpile_regression_datasets:
        raise ValueError(
            f"Dataset {name} not found in the RehabPile collection. "
            "Please check the spelling or run the discovery functions."
        )

    if split not in ["train", "test"]:
        raise ValueError(f"Split must be 'train' or 'test', but found '{split}'.")
    
    # Determine the task and base collection name from the dataset name
    # e.g., "KIMORE_clf_bn_Subject_1" -> "KIMORE_clf_bn", "Subject_1"
    name_parts = name.split("_")
    subfolder = f"{name_parts[-2]}_{name_parts[-1]}" # Handles names like 'Subject_1'
    collection = "_".join(name_parts[:-2])
    
    task_folder = "classification" if "clf" in collection else "regression"

    # Set up the extraction path
    if extract_path is None:
        # Assumes this file is in aeon/datasets, goes up one level
        base_path = Path(__file__).parent / "local_data"
    else:
        base_path = Path(extract_path)
    
    dataset_path = base_path / task_folder / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Define file names and download URLs
    files_to_load = {
        "X": f"X_{split}.npy",
        "y": f"y_{split}.npy",
    }
    
    for file_key, file_name in files_to_load.items():
        local_file_path = dataset_path / file_name
        if not local_file_path.exists():
            print(f"Downloading {file_name} for dataset {name}...")
            file_url = f"{REHABPILE_ROOT_URL}{collection}/{subfolder}/{file_name}"
            try:
                urlretrieve(file_url, local_file_path)
            except HTTPError as e:
                raise HTTPError(
                    f"Failed to download {file_url}. Please check the dataset "
                    f"name and your internet connection. Error: {e}"
                ) from e
    
    # Load data from local .npy files
    X = np.load(dataset_path / files_to_load["X"], allow_pickle=True)
    y = np.load(dataset_path / files_to_load["y"], allow_pickle=True)
    
    return X, y