import tempfile
import zipfile
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union
from urllib.request import urlopen

import numpy as np
import pandas as pd

import aeon
from aeon.datasets.tsad_datasets import tsad_datasets

_DATA_FOLDER = Path(aeon.__file__).parent / "datasets" / "tsad_data"
_TIMEEVAL_BASE_URL = "https://my.hidrive.com/api/sharelink/download?id="
_COLLECTION_DOWNLOAD_IDS = {
    "CalIt2": "QbiGASlZ",
    "Daphnet": "SfCmg30B",
    "Dodgers": "uciGAM1q",
    "Exathlon": "q9imgqn3",
    "Genesis": "kTimAkj5",
    "GHL": "Zkimg1vS",
    "GutenTAG": "j2imAkqU",
    "KDD-TSAD": "8ECmABis",
    "Kitsune": "ICCGgm8F",
    "LTDB": "Csimgc2X",
    "Metro": "YUiGgxao",
    "MGAB": "W1CmA5To",
    "MITDB": "YcCmAEXy",
    "NAB": "dziGgNfN",
    "NASA-MSL": "dPCmgheW",
    "NASA-SMAP": "B6iGA1nA",
    "NormA": "4yCGgwwd",
    "Occupancy": "EBCGgKc2",
    "OPPORTUNITY": "gRiGAW97",
    "SMD": "W0CGA01i",
    "SVDB": "lmCmAjUP",
    "TSB-UAD-artificial": "z8CmAXGx",
    "TSB-UAD-synthetic": "kmimguBY",
    "CATSv2": "WXQc8Zufo",
    "TODS-synthetic": "ewIPtYmmZ",
}


def load_anomaly_detection(
    name: Tuple[str, str],
    split: Literal["train", "test"] = "test",
    extract_path: Optional[PathLike] = None,
    return_metadata: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
]:
    """Load a anomaly detection dataset.

    This function loads TSAD problems into memory, downloading from the TimeEval
    archive (https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html) if
    the data is not available at the specified ``extract_path``. If you want to load a
    problem from a local file, specify the location in ``extract_path``. This function
    assumes the data is stored in the TimeEval format

    ...

    If you do not specify ``extract_path``, it will set the path to
    ``aeon/datasets/tsad_data``. If the problem is not present in ``extract_path``, it
    will attempt to download the data.

    The problem name is a tuple of collection name and dataset name.
    ``("KDD-TSAD", "001_UCR_Anomaly_DISTORTED1sddb40")`` is an example of a univariate
    unsupervised problem, ``("CATSv2", "CATSv2")`` a multivariate supervised problem.

    Parameters
    ----------
    name : tuple of str, str
        Name of dataset. If a dataset that is listed in tsad_datasets is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from the TimeEval archive saving it to
        the extract_path.
    split : str{"train", "test"}, default="test"
        Whether to load the train or test partition of the problem. By default, it
        loads the test partition.
    extract_path : str, default=None
        The path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/tsad_data/`. If a path is given, it can be absolute,
        e.g., C:/Temp/ or relative, e.g. Temp/ or ./Temp/.
    return_metadata : boolean, default = False
        If True, returns a tuple (X, y, metadata).

    Returns
    -------
    X: np.ndarray
        The univariate (1d) or multivariate (2d) time series with shape
        (n_instances, n_channels).
    y: numpy array
        The binary anomaly labels with shape (n_instances,).
    metadata: optional
        returns the following metadata
        'problemname',timestamps,dimensions,learning_type,contamination,num_anomalies

    Raises
    ------
    URLError or HTTPError
        If the website is not accessible.
    ValueError
        If a dataset name that does not exist on the repo is given or if a
        dataset is requested that does not exist in the archive.

    Examples
    --------
    >>> from aeon.datasets import load_anomaly_detection
    >>> X, y = load_anomaly_detection(
    ...     name=("KDD-TSAD", "001_UCR_Anomaly_DISTORTED1sddb40")
    ... )  # doctest: +SKIP
    """
    if not isinstance(name, tuple) or len(name) != 2:
        raise ValueError(
            "The name of the dataset must be a tuple of two strings specifying dataset "
            "collection and dataset name, e.g., "
            "('KDD-TSAD', '001_UCR_Anomaly_DISTORTED1sddb40')"
        )
    if extract_path is not None:
        data_folder = Path(extract_path)
    else:
        data_folder = _DATA_FOLDER
    if not data_folder.exists():
        data_folder.mkdir(parents=True)

    # Check if the dataset is part of the TimeEval archive
    if name not in tsad_datasets:
        return _load_custom(name, split, data_folder, return_metadata)

    # Load index
    df_meta = pd.read_csv(data_folder / "datasets.csv")
    df_meta = df_meta.set_index(["collection_name", "dataset_name"])
    metadata = df_meta.loc[name]
    if split == "train":
        if metadata["train_path"] is None or np.isnan(metadata["train_path"]):
            raise ValueError(
                f"Dataset {name} does not have a training partition. Only "
                "`split='test'` is supported."
            )
        dataset_path = data_folder / metadata["train_path"]
    else:
        dataset_path = data_folder / metadata["test_path"]

    if not dataset_path.exists():
        # attempt download, always downloads the full collection (ZIP-file)
        _download_and_extract(name[0], data_folder)

    # Load the data
    X, y = _load_saved_dataset(
        dataset_path, squeeze=metadata["input_type"] == "univariate"
    )

    if return_metadata:
        meta = {
            "problemname": " ".join(name),
            "timestamps": metadata["length"],
            "dimensions": metadata["dimensions"],
            "learning_type": metadata["train_type"],
            "contamination": metadata["contamination"],
            "num_anomalies": metadata["num_anomalies"],
        }
        return X, y, meta
    return X, y


def _load_custom(
    name: Tuple[str, str],
    split: Literal["train", "test"],
    path: Path,
    return_metadata: bool,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
]:
    if not path.is_file() or path.suffix != ".csv":
        raise ValueError(
            "When loading a custom dataset, the extract_path must point to a "
            f"TimeEval-formatted CSV file, but {path} is not a CSV-file."
        )
    X, y = _load_saved_dataset(path)
    if return_metadata:
        learning_type = "unsupervised"
        if split == "train":
            learning_type = "semi-supervised" if np.sum(y) == 0 else "supervised"
        meta = {
            "problemname": " ".join(name),
            "timestamps": X.shape[0],
            "dimensions": X.shape[1] if len(X.shape) > 1 else 1,
            "learning_type": learning_type,
            "contamination": np.sum(y) / len(y),
            "num_anomalies": int(np.sum(np.diff(np.r_[0, y, 0]) == -1)),
        }
        return X, y, meta
    return X, y


def _download_and_extract(collection: str, extract_path: Path) -> None:
    if collection not in _COLLECTION_DOWNLOAD_IDS:
        raise ValueError(
            f"Collection {collection} is part of the TimeEval archive but not "
            "available for download (missing permission to share)."
        )

    url = _TIMEEVAL_BASE_URL + _COLLECTION_DOWNLOAD_IDS[collection]

    with tempfile.TemporaryDirectory() as dl_dir:
        zip_file_name = Path(dl_dir) / f"{collection}.zip"
        # Using urlopen instead of urlretrieve
        with urlopen(url, timeout=60) as response:
            with open(zip_file_name, "wb") as out_file:
                out_file.write(response.read())

        try:
            with zipfile.ZipFile(zip_file_name, "r") as fh:
                fh.extractall(extract_path)
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(
                "Could not unzip dataset. Please make sure the URL is valid."
            )


def _load_saved_dataset(
    path: Path, squeeze: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:-1].values
    if squeeze:
        X = X.ravel()
    y = df.iloc[-1].values
    return X, y
