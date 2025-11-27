"""Dataset loading functions for Monster datasets."""

__maintainer__ = []
__all__ = [
    "load_monster_dataset_names",
    "load_monster_dataset",
]

import numpy as np

from aeon.utils.numba.general import z_normalise_series_3d
from aeon.utils.validation._dependencies import _check_soft_dependencies

ORG_ID = "monster-monash"

_monster_dataset_names = None


def _fetch_monster_dataset_names() -> list[str]:
    """Fetch the list of Monster dataset names from Hugging Face Hub."""
    _check_soft_dependencies("huggingface-hub", severity="error")
    from huggingface_hub import list_datasets

    datasets = list_datasets(author=ORG_ID)
    dataset_names = []
    for dataset_info in datasets:
        if dataset_info.id.startswith(f"{ORG_ID}/"):
            name = dataset_info.id.split("/")[-1]
            dataset_names.append(name)

    return sorted(dataset_names)


def _lazy_load_monster_names():
    """Fetch and cache names, but only on the first call."""
    global _monster_dataset_names
    if _monster_dataset_names is None:
        _monster_dataset_names = _fetch_monster_dataset_names()


def load_monster_dataset_names() -> list[str]:
    """Load the list of available Monster dataset names from Hugging Face Hub.

    Returns
    -------
    list of str
        A list of available Monster dataset names.
    """
    _lazy_load_monster_names()
    return _monster_dataset_names


def load_monster_dataset(
    dataset_name: str,
    fold: int = 0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a Monster dataset from Hugging Face Hub.

     MONSTER— the MONash Scalable Time Series Evaluation Repository,
     introduced in [1]_, is a collection of large datasets for time
     series classification.The collection is hosted on Hugging Face Hub.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load (e.g., "CornellWhaleChallenge", "AudioMNIST").
    fold : int, default=0
        The specific cross-validation fold index to load. This determines which
        samples are used for the test set. Defaults to fold 0.
    normalize : bool, default=True
        If True, the time series data (X) is Z-normalized (mean=0, std=1) across
        the series length using `z_normalise_series_3d`.

    Returns
    -------
    X_train : np.ndarray
        The training data, shape (n_train_cases, n_channels, n_timepoints).
        (n_channels=1 for these univariate datasets).
    y_train : np.ndarray
        The training class labels, shape (n_train_cases,).
    X_test : np.ndarray
        The testing data, shape (n_test_cases, n_channels, n_timepoints).
    y_test : np.ndarray
        The testing class labels, shape (n_test_cases,).

    Raises
    ------
    ModuleNotFoundError
        If required optional dependency 'huggingface-hub' not installed.
    ValueError
        If the `dataset_name` is not recognized
        or the `fold` number is invalid.
    OSError
        If the download fails due to network issues

    Notes
    -----
    The data files are cached locally by the `huggingface-hub`
    library, avoiding repeated downloads. This function
    requires the optional dependency `huggingface-hub`.

    References
    ----------
    .. [1] Dempster, A., Mohammadi Foumani, N., Tan, C. W., Miller,
        L., Mishra, A., Salehi, M., Pelletier, C., Schmidt, D. F.,
        & Webb, G. I. (2025). MONSTER: Monash Scalable
        Time Series Evaluation Repository. arXiv preprint arXiv:2502.15122.

    """
    _check_soft_dependencies("huggingface-hub", severity="error")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubDownloadError

    repo_id = f"{ORG_ID}/{dataset_name}"

    if dataset_name not in load_monster_dataset_names():
        raise ValueError(f"Dataset {dataset_name} not found in the Monster collection.")

    data_path = hf_hub_download(
        repo_id=repo_id, filename=f"{dataset_name}_X.npy", repo_type="dataset"
    )
    X = np.load(data_path, mmap_mode="r")
    if normalize:
        X = z_normalise_series_3d(X)

    label_filename = f"{dataset_name}_Y.npy"
    try:
        label_path = hf_hub_download(
            repo_id=repo_id, filename=label_filename, repo_type="dataset"
        )
    except HfHubDownloadError:
        label_filename = f"{dataset_name}_y.npy"
        label_path = hf_hub_download(
            repo_id=repo_id, filename=label_filename, repo_type="dataset"
        )
    y = np.load(label_path)

    try:
        test_index_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"test_indices_fold_{fold}.txt",
            repo_type="dataset",
        )
        test_index = np.loadtxt(test_index_path, dtype=int)
    except Exception as e:
        raise OSError(f"Failed to load test indices for fold {fold}: {e}. ") from e

    test_bool_index = np.zeros(len(y), dtype=bool)
    test_bool_index[test_index] = True
    return (
        X[~test_bool_index],
        y[~test_bool_index],
        X[test_bool_index],
        y[test_bool_index],
    )
