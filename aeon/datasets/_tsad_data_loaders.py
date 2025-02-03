"""Dataset loading functions for anomaly detection."""

__all__ = [
    "load_anomaly_detection",
    "load_from_timeeval_csv_file",
    "load_kdd_tsad_135",
    "load_daphnet_s06r02e0",
    "load_ecg_diff_count_3",
]

import tempfile
import zipfile
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Optional, Union
from urllib.request import urlopen

import numpy as np
import pandas as pd

import aeon
from aeon.datasets.tsad_datasets import _load_indexfile, tsad_datasets

_DATA_FOLDER = Path(aeon.__file__).parent / "datasets" / "local_data"
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
    name: tuple[str, str],
    split: Literal["train", "test"] = "test",
    extract_path: Optional[PathLike] = None,
    return_metadata: bool = False,
) -> Union[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, dict[str, Any]]
]:
    """Load an anomaly detection dataset.

    This function loads TSAD problems into memory, downloading from the TimeEval
    archive (https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html) [1] if
    the data is not available at the specified ``extract_path``. If you want to load a
    problem from a local file, specify the location in ``extract_path``. This function
    assumes the data is stored in the
    `TimeEval format <https://timeeval.readthedocs.io/en/latest/concepts/datasets.html
    #canonical-file-format>`_.

    If you do not specify ``extract_path``, it will set the path to
    ``aeon/datasets/local_data``. If the problem is not present in ``extract_path``, it
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
        looks in `aeon/datasets/local_data/`. If a path is given, it can be an absolute,
        e.g., C:/Temp/ or relative, e.g. Temp/ or ./Temp/, path to an existing CSV-file.
    return_metadata : boolean, default = False
        If True, returns a tuple (X, y, metadata).

    Returns
    -------
    X: np.ndarray
        The univariate (1d) or multivariate (2d) time series with shape
        (n_instances, n_channels).
    y: np.ndarray
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

    References
    ----------
    .. [1] Sebastian Schmidl, Phillip Wenig, Thorsten Papenbrock: Anomaly Detection in
           Time Series: A Comprehensive Evaluation. PVLDB 9:(15), 2022,
           DOI:10.14778/3538598.3538602.
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

    # Check if the dataset is part of the TimeEval archive
    if name not in tsad_datasets():
        return _load_custom(name, split, data_folder, return_metadata)

    # Load index
    df_meta = _load_indexfile()
    df_meta = df_meta.set_index(["collection_name", "dataset_name"])
    metadata = df_meta.loc[name]
    if split.lower() == "train":
        train_path = metadata["train_path"]
        if train_path is None or pd.isnull(train_path):
            raise ValueError(
                f"Dataset {name} does not have a training partition. Only "
                "`split='test'` is supported."
            )
        dataset_path = data_folder / train_path
    else:
        dataset_path = data_folder / metadata["test_path"]

    if not dataset_path.exists():
        # attempt download, always downloads the full collection (ZIP-file)
        _download_and_extract(name[0], data_folder)

    # Load the data
    X, y = load_from_timeeval_csv_file(dataset_path)

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
    name: tuple[str, str],
    split: Literal["train", "test"],
    path: Path,
    return_metadata: bool,
) -> Union[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, dict[str, Any]]
]:
    if not path.is_file() or path.suffix != ".csv":
        raise ValueError(
            "When loading a custom dataset, the extract_path must point to a "
            f"TimeEval-formatted CSV file, but {path} is not a CSV-file."
        )
    X, y = load_from_timeeval_csv_file(path)
    if return_metadata:
        learning_type = "unsupervised"
        if split.lower() == "train":
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
            # e.g. SSA, IOPS, WebscopeS5
        )

    url = _TIMEEVAL_BASE_URL + _COLLECTION_DOWNLOAD_IDS[collection]

    with tempfile.TemporaryDirectory() as dl_dir:
        zip_file_name = Path(dl_dir) / f"{collection}.zip"
        # Using urlopen instead of urlretrieve
        with urlopen(url, timeout=60) as response:
            with open(zip_file_name, "wb") as out_file:
                out_file.write(response.read())

        try:
            extract_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file_name, "r") as fh:
                fh.extractall(extract_path)
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(
                "Could not unzip dataset. Please make sure the URL is valid."
            )


def load_from_timeeval_csv_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a TimeEval-formatted CSV file into memory.

    TimeEval datasets are stored in simple CSV files with the following format:

    - The first column contains the timestamps; usually with the header 'timestamp'.
    - The last column contains the binary anomaly labels; usually with the header
      'is_anomaly'.
    - The columns in between contain the time series data.

    This function does not parse the timestamp information because the data is expected
    to be equidistant and has no missing values.

    Parameters
    ----------
    path : Path
        Full or relative path to the CSV file.

    Returns
    -------
    X : np.ndarray
        The univariate (1d) or multivariate (2d) time series with shape
        (n_instances, n_channels). If univariate the shape is (n_instances,).
    y : np.ndarray
        The binary anomaly labels with shape (n_instances,).
    """
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1].values
    if X.ndim == 2 and X.shape[1] == 1:
        X = X.ravel()
    y = df.iloc[:, -1].values
    return X, y


def load_kdd_tsad_135(
    split: Literal["train", "test"] = "test",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the KDD-TSAD 135 UCR_Anomaly_Internal_Bleeding16 univariate dataset.

    Returns
    -------
    X : np.ndarray
        Univariate time series with shape (n_timepoints,).
    y : np.ndarray
        Binary anomaly labels with shape (n_timepoints,).

    Examples
    --------
    >>> from aeon.datasets import load_kdd_tsad_135
    >>> X, y = load_kdd_tsad_135()

    Notes
    -----
    The KDD-TSAD collection contains 250 datasets from different sources with a single
    anomaly annotated in each dataset (the top discord). It was prepared for a
    competition at SIGKDD 2021 [1]. Dataset 135 is a univariate dataset from this
    collection that records the heart rate of a patient with internal bleeding.

        Dimensionality:     univariate
        Series length:      7501
        Learning Type:      semi-supervised (normal training data)


    References
    ----------
    .. [1] Keogh, E., Dutta Roy, T., Naik, U. & Agrawal, A (2021). Multi-dataset
           Time-Series Anomaly Detection Competition, SIGKDD 2021.
           https://compete.hexagon-ml.com/practice/competition/39/
    """
    # name = ("KDD-TSAD", "135_UCR_Anomaly_InternalBleeding16")
    dataset_path = (
        Path(aeon.__file__).parent
        / "datasets"
        / "data"
        / "KDD-TSAD_135"
        / f"135_UCR_Anomaly_InternalBleeding16_{split.upper()}.csv"
    )
    X, y = load_from_timeeval_csv_file(dataset_path)
    return X, y


def load_daphnet_s06r02e0() -> tuple[np.ndarray, np.ndarray]:
    """Load the Daphnet S06R02E0 multivariate time series dataset.

    Returns
    -------
    X : np.ndarray
        Multivariate time series with shape (28800,9).
    y : np.ndarray
        Binary anomaly labels with shape (28800,).

    Examples
    --------
    >>> from aeon.datasets import load_daphnet_s06r02e0
    >>> X, y = load_daphnet_s06r02e0()

    Notes
    -----
    The Daphnet collection [1] contains unsupervised datasets with multivariate time
    series. Each dataset contains the annotated readings of 3 acceleration sensors at
    the hip and leg of Parkinson's disease patients that experience freezing of gait
    (FoG) during walking tasks. FoG occureances are annotated as anomalies.

        Dimensionality:     multivariate (9 channels)
        Series length:      7040
        Frequency:          around 256 Hz
        Learning Type:      unsupervised (no training data)


    References
    ----------
    .. [1] Roggen, Daniel, Plotnik, Meir, and Hausdorff,Jeff. (2013). Daphnet Freezing
           of Gait. UCI Machine Learning Repository. https://doi.org/10.24432/C56K78.
    """
    # name = ("Daphnet", "S06R02E0")
    dataset_path = (
        Path(aeon.__file__).parent
        / "datasets"
        / "data"
        / "Daphnet_S06R02E0"
        / "S06R02E0.csv"
    )
    X, y = load_from_timeeval_csv_file(dataset_path)
    return X, y


def load_ecg_diff_count_3(
    learning_type: Literal[
        "unsupervised", "semi-supervised", "supervised"
    ] = "unsupervised",
) -> Union[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Load the synthetic ECG dataset 'ecg-diff-count-3'.

    The dataset contains three different kind of anomalies. The dataset was generated
    using `GutenTAG <https://github.com/TimeEval/gutentag>`_ [1]

    Parameters
    ----------
    learning_type : str, default = "unsupervised"
        The learning type of the dataset. Must be one of "unsupervised",
        "semi-supervised", or "supervised". If "unsupervised", only the test partition
        is loaded. If "semi-supervised", the test partition and the training partition
        **without** anomalies is returned. If "supervised", the training partition
        **with** anomalies is returned instead.

    Returns
    -------
    X_test : np.ndarray
        Multivariate test time series with shape (10000,2).
    y_test : np.ndarray
        Binary anomaly labels for the test time series with shape (10000,).
    X_train : np.ndarray, optional
        Multivariate train time series with shape (10000,2). Omitted if
        ``learning_type`` is "unsupervised".
    y_train : np.ndarray, optional
        Binary anomaly labels for the train time series with shape (10000,). Omitted if
        ``learning_type`` is "unsupervised".

    Examples
    --------
    >>> from aeon.datasets import load_ecg_diff_count_3
    >>> X_test, y_test, X_train, y_train = load_ecg_diff_count_3("supervised")

    Notes
    -----
        Dimensionality:     univariate
        Series length:      10000
        Frequency:          unknown
        Learning Type:      all supported

    References
    ----------
    .. [1] Phillip Wenig, Sebastian Schmidl, and Thorsten Papenbrock. TimeEval: A
           Benchmarking Toolkit for Time Series Anomaly Detection Algorithms. PVLDB,
           15(12): 3678 - 3681, 2022. doi:10.14778/3554821.3554873.
    """
    # name = ("correlation-anomalies", "ecg-diff-count-3")
    base_path = Path(aeon.__file__).parent / "datasets" / "data" / "UnitTest"
    test_file = base_path / "ecg-diff-count-3_TEST.csv"
    train_semi_file = base_path / "ecg-diff-count-3_TRAIN_NA.csv"
    train_super_file = base_path / "ecg-diff-count-3_TRAIN_A.csv"

    X_test, y_test = load_from_timeeval_csv_file(test_file)
    if learning_type == "semi-supervised":
        X_train, y_train = load_from_timeeval_csv_file(train_semi_file)
        return X_test, y_test, X_train, y_train
    if learning_type == "supervised":
        X_train, y_train = load_from_timeeval_csv_file(train_super_file)
        return X_test, y_test, X_train, y_train
    return X_test, y_test
