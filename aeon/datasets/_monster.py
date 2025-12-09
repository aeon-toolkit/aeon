"""Load functions for the MONSTER datasets without huggingface_hub dependency."""

import os
import urllib.request
from urllib.error import HTTPError

import numpy as np

import aeon
from aeon.utils.numba.general import z_normalise_series_3d

__all__ = ["load_monster"]

# List of available datasets for reference
monash_monster_datasets = [
    "CornellWhaleChallenge",
    "AudioMNIST",
    "WhaleSounds",
    "Pedestrian",
    "FruitFlies",
    "AudioMNIST-DS",
    "Traffic",
    "LakeIce",
    "MosquitoSound",
    "InsectSound",
]


def _download_from_hf(repo_id: str, filename: str, local_dir: str) -> str:
    """Download a raw file from Hugging Face via HTTPS.

    This avoids the need for the huggingface_hub library.
    """
    # Construct the raw URL
    # 'resolve/main' gets the file from the main branch
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

    local_path = os.path.join(local_dir, filename)

    # If file already exists, don't download again (Basic Caching)
    if os.path.exists(local_path):
        return local_path

    # Ensure directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        # We removed the print() statement here to satisfy flake8 T201
        urllib.request.urlretrieve(url, local_path)
    except HTTPError as e:
        # Clean up empty file if download fails
        if os.path.exists(local_path):
            os.remove(local_path)
        raise ValueError(
            f"Could not download {filename}. Check dataset name or internet connection."
        ) from e

    return local_path


def load_monster(
    name: str, fold: int = 0, normalize: bool = True, extract_path: str = None
):
    """
    Load a MONSTER dataset from Hugging Face using raw URLs.

    Parameters
    ----------
    name : str
        Name of the dataset to load (e.g., 'AudioMNIST').
    fold : int, default=0
        The fold number for the test/train split.
    normalize : bool, default=True
        If True, z-normalize the time series data.
    extract_path : str, optional
        Path to save downloaded data. If None, uses aeon/datasets/local_data.

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """
    # Define where to save the data
    if extract_path is None:
        # Use the standard aeon data location
        module_path = os.path.dirname(aeon.__file__)
        extract_path = os.path.join(module_path, "datasets", "local_data", name)

    repo_id = f"monster-monash/{name}"

    # 1. Download X (Data)
    try:
        data_path = _download_from_hf(repo_id, f"{name}_X.npy", extract_path)
    except ValueError:
        raise ValueError(f"Dataset '{name}' not found in monster-monash repository.")

    # Load using memory map to handle large files efficiently
    X = np.load(data_path, mmap_mode="r")

    if normalize:
        # Note: Normalizing mmap data might load it into RAM.
        # Be careful if data is bigger than RAM.
        X = z_normalise_series_3d(X)

    # 2. Download Y (Labels) - Handle Case Sensitivity
    # Some files on HF are named _Y.npy, others _y.npy
    try:
        label_path = _download_from_hf(repo_id, f"{name}_Y.npy", extract_path)
    except ValueError:
        try:
            label_path = _download_from_hf(repo_id, f"{name}_y.npy", extract_path)
        except ValueError as e:
            raise ValueError(f"Could not find labels for {name}") from e

    y = np.load(label_path)

    # 3. Download Test Indices
    try:
        test_index_path = _download_from_hf(
            repo_id, f"test_indices_fold_{fold}.txt", extract_path
        )
        test_index = np.loadtxt(test_index_path, dtype=int)
    except ValueError:
        raise ValueError(f"Failed to load test indices for fold {fold}")

    # 4. Create Split
    test_bool_index = np.zeros(len(y), dtype=bool)
    test_bool_index[test_index] = True

    X_train = X[~test_bool_index]
    y_train = y[~test_bool_index]
    X_test = X[test_bool_index]
    y_test = y[test_bool_index]

    return X_train, y_train, X_test, y_test