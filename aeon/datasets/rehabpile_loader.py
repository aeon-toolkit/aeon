"""Dataset loading functions for RehabPile dataset."""

import json
import re
import warnings
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

# The root URL for the RehabPile dataset collection
REHABPILE_ROOT_URL = "https://maxime-devanne.com/datasets/RehabPile/"

# The list of base collections as defined in the script
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
# REHABPILE DATASET FOLDS
REHABPILE_FOLDS = {
    "classification": {
        "IRDS_clf_bn_EFL": 5,
        "IRDS_clf_bn_EFR": 5,
        "IRDS_clf_bn_SAL": 5,
        "IRDS_clf_bn_SAR": 5,
        "IRDS_clf_bn_SFE": 5,
        "IRDS_clf_bn_SFL": 5,
        "IRDS_clf_bn_SFR": 5,
        "IRDS_clf_bn_STL": 5,
        "IRDS_clf_bn_STR": 5,
        "KERAAL_clf_bn_CTK": 6,
        "KERAAL_clf_bn_ELK": 6,
        "KERAAL_clf_bn_RTK": 6,
        "KERAAL_clf_mc_CTK": 6,
        "KERAAL_clf_mc_ELK": 6,
        "KERAAL_clf_mc_RTK": 6,
        "KIMORE_clf_bn_LA": 5,
        "KIMORE_clf_bn_LT": 5,
        "KIMORE_clf_bn_PR": 5,
        "KIMORE_clf_bn_Sq": 5,
        "KIMORE_clf_bn_TR": 5,
        "KINECAL_clf_bn_3WFV": 5,
        "KINECAL_clf_bn_GGFV": 5,
        "KINECAL_clf_bn_QSEC": 5,
        "KINECAL_clf_bn_QSEO": 5,
        "SPHERE_clf_bn_WUS": 6,
        "UCDHE_clf_bn_MP": 5,
        "UCDHE_clf_bn_Rowing": 5,
        "UCDHE_clf_mc_MP": 5,
        "UCDHE_clf_mc_Rowing": 5,
        "UIPRMD_clf_bn_DS": 5,
        "UIPRMD_clf_bn_HS": 7,
        "UIPRMD_clf_bn_IL": 6,
        "UIPRMD_clf_bn_SASLR": 8,
        "UIPRMD_clf_bn_SL": 8,
        "UIPRMD_clf_bn_SSA": 7,
        "UIPRMD_clf_bn_SSE": 8,
        "UIPRMD_clf_bn_SSIER": 7,
        "UIPRMD_clf_bn_SSS": 6,
        "UIPRMD_clf_bn_STS": 5,
    },
    "regression": {
        "EHE_reg_BWL": 5,
        "EHE_reg_BWR": 5,
        "EHE_reg_HUD": 5,
        "EHE_reg_WB": 5,
        "EHE_reg_WF": 5,
        "EHE_reg_WH": 5,
        "KIMORE_reg_LA": 5,
        "KIMORE_reg_LT": 5,
        "KIMORE_reg_PR": 5,
        "KIMORE_reg_Sq": 5,
        "KIMORE_reg_TR": 5,
        "UIPRMD_reg_DS": 5,
        "UIPRMD_reg_HS": 7,
        "UIPRMD_reg_IL": 6,
        "UIPRMD_reg_SASLR": 8,
        "UIPRMD_reg_SL": 8,
        "UIPRMD_reg_SSA": 7,
        "UIPRMD_reg_SSE": 8,
        "UIPRMD_reg_SSIER": 7,
        "UIPRMD_reg_SSS": 6,
        "UIPRMD_reg_STS": 5,
    },
}

_rehabpile_classification_datasets = None
_rehabpile_regression_datasets = None


def _fetch_rehabpile_dataset_names() -> tuple[list[str], list[str]]:
    """Scrape the RehabPile website to discover all available datasets.

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
            warnings.warn(
                f"Could not access {collection_url}. Skipping. Error: {e}", stacklevel=2
            )
            continue

        subfolders = re.findall(r'href="([^"/]+)/"', html_content)

        for folder in subfolders:
            dataset_name = f"{collection}_{folder}"
            if "clf" in collection:
                classification_datasets.append(dataset_name)
            elif "reg" in collection:
                regression_datasets.append(dataset_name)

    return sorted(classification_datasets), sorted(regression_datasets)


def _lazy_load_rehabpile_names():
    """Fetch and cache names, but only on the first call."""
    global _rehabpile_classification_datasets, _rehabpile_regression_datasets
    if _rehabpile_classification_datasets is None:
        (
            _rehabpile_classification_datasets,
            _rehabpile_regression_datasets,
        ) = _fetch_rehabpile_dataset_names()


def load_rehab_pile_classification_datasets() -> list[str]:
    """Return a list of all classification datasets in the RehabPile archive.

    Returns
    -------
    list[str]
        A list of dataset names for the classification task.
    """
    _lazy_load_rehabpile_names()
    return _rehabpile_classification_datasets


def load_rehab_pile_regression_datasets() -> list[str]:
    """Return a list of all regression datasets in the RehabPile archive.

    Returns
    -------
    list[str]
        A list of dataset names for the regression task.
    """
    _lazy_load_rehabpile_names()
    return _rehabpile_regression_datasets


def load_rehab_pile_dataset(
    name: str,
    split: Literal["train", "test"] = "train",
    fold: int = 0,
    extract_path: str | Path | None = None,
    return_meta: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a dataset from the RehabPile collection.

    The RehabPile collection, introduced in [1]_, is a unified archive of existing
    rehabilitation datasets designed to serve as a benchmark for automated human
    motion assessment. Unlike general human activity recognition, the goal of
    rehabilitation assessment is to analyze the quality of movement within the
    same action class, requiring the detection of subtle deviations from an
    ideal motion.

    The collection is composed of 8 original repositories, compiled into 39
    classification and 21 regression problems. The data consists of skeleton-based
    human motion sequences captured from video streams or inertial sensors. Each
    problem is divided into multiple cross-validation folds to ensure robust
    evaluation.

    Parameters
    ----------
    name : str
        The name of the RehabPile dataset, e.g., "KIMORE_clf_bn_TR".
        A full list can be obtained from `load_rehab_pile_classification_datasets()`
        or `load_rehab_pile_regression_datasets()`.
    split : {"train", "test"}, default="train"
        The split of the data to return.
    fold : int, default=0
        The cross-validation fold (resample) to load. Defaults to the first fold (0).
        The number of available folds varies by dataset.
    extract_path : str or Path, default=None
        The path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/local_data/`.
    return_meta : bool, default=False
        If True, returns a tuple (X, y, meta_data). The meta_data is a dictionary
        containing information about the dataset, loaded from the `info.json` file.

    Returns
    -------
    X : np.ndarray
        The time series data of shape (n_cases, n_channels, n_timepoints).
    y : np.ndarray
        The target values (class labels or regression values) of shape (n_cases,).
    meta_data : dict, optional
        A dictionary containing metadata for the dataset, such as the number of
        classes, channels, and series length. Only returned if `return_meta` is True.

    Notes
    -----
    webpage: https://msd-irimas.github.io/pages/DeepRehabPile/#rehab-pile


    References
    ----------
    .. [1] Ismail-Fawaz, Ali, Maxime Devanne, Stefano Berretti, Jonathan Weber,
           and Germain Forestier."Deep Learning for Skeleton Based Human Motion
           Rehabilitation Assessment:A Benchmark." arXiv preprint
           arXiv:2507.21018 (2025).

    """
    task_folder = "classification" if "clf" in name else "regression"
    if (
        name not in load_rehab_pile_classification_datasets()
        and name not in load_rehab_pile_regression_datasets()
    ):
        raise ValueError(
            f"Dataset {name} not found in the RehabPile collection. "
            "Please check the spelling or run the discovery functions."
        )

    if split not in ["train", "test"]:
        raise ValueError(f"Split must be 'train' or 'test', but found '{split}'.")

    num_folds = REHABPILE_FOLDS[task_folder].get(name)
    if num_folds is None:
        raise ValueError(f"Fold information for dataset {name} not found.")
    if not 0 <= fold < num_folds:
        raise ValueError(
            f"Invalid fold={fold} for dataset {name}. "
            f"Valid folds are 0 to {num_folds - 1}."
        )

    name_parts = name.split("_")
    subfolder = name_parts[-1]
    collection = "_".join(name_parts[:-1])

    if extract_path is None:
        base_path = Path(__file__).parent / "local_data"
    else:
        base_path = Path(extract_path)

    dataset_path_fold = base_path / task_folder / name / str(fold)
    dataset_path_fold.mkdir(parents=True, exist_ok=True)
    dataset_path_meta = base_path / task_folder / name
    dataset_path_meta.mkdir(parents=True, exist_ok=True)

    files_to_load = {
        "X": f"x_{split}_fold{fold}.npy",
        "y": f"y_{split}_fold{fold}.npy",
    }

    for _, file_name in files_to_load.items():
        local_file_path = dataset_path_fold / file_name
        if not local_file_path.exists():
            file_url = (
                f"{REHABPILE_ROOT_URL}{collection}/{subfolder}/fold{fold}/{file_name}"
            )
            try:
                with urlopen(file_url, timeout=60) as response:
                    with open(local_file_path, "wb") as out_file:
                        out_file.write(response.read())
            except (HTTPError, URLError, TimeoutError) as e:
                raise OSError(
                    f"Failed to download {file_url}. Please check the dataset "
                    f"name and your internet connection. Error: {e}"
                ) from e

    X = np.load(dataset_path_fold / files_to_load["X"], allow_pickle=True)
    y = np.load(dataset_path_fold / files_to_load["y"], allow_pickle=True)

    # Correct issue with pickle loading, ensuring all items are numpy arrays

    X = np.array(X.tolist())
    y = np.array(y.tolist())

    if return_meta:
        meta_path = dataset_path_meta / "info.json"
        if not meta_path.exists():
            meta_url = f"{REHABPILE_ROOT_URL}{collection}/{subfolder}/info.json"
            try:
                with urlopen(meta_url, timeout=60) as response:
                    with open(meta_path, "wb") as out_file:
                        out_file.write(response.read())
            except (HTTPError, URLError, TimeoutError) as e:
                raise OSError(
                    f"Failed to download metadata from {meta_url}. " f"Error: {e}"
                ) from e
        with open(meta_path) as f:
            meta_data = json.load(f)
        return X, y, meta_data

    return X, y
