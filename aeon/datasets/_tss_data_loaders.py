"""Dataset loading functions for segmentation."""

__all__ = [
    "load_time_series_segmentation_benchmark",
    "load_human_activity_segmentation_datasets",
]

from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import aeon

_DATA_FOLDER = Path(aeon.__file__).parent / "datasets" / "local_data"
_TSSB_URL = (
    "https://raw.githubusercontent.com/ermshaua/time-series-segmentation"
    "-benchmark/main/tssb/datasets/tssb.csv.zip"
)
_HAS_URL = (
    "https://raw.githubusercontent.com/patrickzib"
    "/human_activity_segmentation_challenge/main/datasets/has2023_master.csv"
    ".zip"
)


def load_time_series_segmentation_benchmark(
    extract_path: Optional[PathLike] = None,
    return_metadata: bool = False,
) -> Union[
    tuple[list[np.ndarray], list[np.ndarray]],
    tuple[list[np.ndarray], list[np.ndarray], list[tuple[str, int]]],
]:
    """Load the Time Series Segmentation Benchmark (TSSB).

    This function loads the Time Series Segmentation Benchmark (TSSB) into memory,
    downloading from GitHub (https://github.com/ermshaua/time-series-segmentation
    -benchmark) [1] if the data is not available at the specified ``extract_path``.
    The benchmark contains 75 annotated TS with 1-9 segments. Each TS is constructed
    from one of the UEA & UCR time series classification datasets. TS are grouped by
    label and concatenated to create segments with distinctive temporal patterns and
    statistical properties. Offsets at which segments change are annotated as CPs.
    Addtionally, resampling  is applied to control the data resolution. Approximate,
    hand-selected window sizes are provided that capture temporal patterns.

    If you do not specify ``extract_path``, it will set the path to
    ``aeon/datasets/local_data``. If the problem is not present in ``extract_path``, it
    will attempt to download the data.

    Parameters
    ----------
    extract_path : str, default=None
        The path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/local_data/`. If a path is given, it can be an absolute,
        e.g., C:/Temp/ or relative, e.g. Temp/ or ./Temp/, path to an existing CSV-file.
    return_metadata : boolean, default = False
        If True, returns a tuple (X, y, metadata).

    Returns
    -------
    X: list of np.ndarray
        The list of univariate (1d) time series with variable shape (n_instances,).
    y: list of np.ndarray
        The list of change points for every time series.
    metadata: optional
        The list of tuples containing data set names and window sizes

    Raises
    ------
    URLError or HTTPError
        If the GitHub repository is not accessible.

    Examples
    --------
    >>> from aeon.datasets import load_time_series_segmentation_benchmark
    >>> X, y = load_time_series_segmentation_benchmark()
    ... )  # doctest: +SKIP

    References
    ----------
    .. [1] Arik Ermshaus, Patrick Schäfer, Ulf Leser: ClaSP: parameter-free
           time series segmentation. Data Mining and Knowledge Discovery, 2023,
           DOI:10.1007/s10618-023-00923-x.
    """
    # set default/custom data folder
    if extract_path is not None:
        data_folder = Path(extract_path)
    else:
        data_folder = _DATA_FOLDER

    benchmark_path = _DATA_FOLDER / "tssb.csv"

    # converters to correctly load benchmark
    np_cols = ["change_points", "time_series"]
    converters = {col: lambda val: np.array(eval(val)) for col in np_cols}

    # load benchmark from git repo (and save locally) / or load locally
    if not benchmark_path.exists():
        data_folder.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(_TSSB_URL, converters=converters, compression="zip")

        # make sure numerical data is correctly saved
        for np_col in np_cols:
            df[np_col] = df[np_col].apply(np.ndarray.tolist)
        df.to_csv(benchmark_path, index=None)

    df = pd.read_csv(benchmark_path, converters=converters)

    # construct return data
    X = df.time_series.tolist()
    y = df.change_points.tolist()

    # construct meta data
    if return_metadata is True:
        metadata = [tuple(row) for _, row in df[["dataset", "window_size"]].iterrows()]
        return X, y, metadata

    return X, y


def load_human_activity_segmentation_datasets(
    extract_path: Optional[PathLike] = None,
    return_metadata: bool = False,
) -> Union[
    tuple[list[np.ndarray], list[np.ndarray]],
    tuple[
        list[np.ndarray], list[np.ndarray], list[tuple[str, str, int, int, np.ndarray]]
    ],
]:
    """Load the Human Activity Segmentation Challenge data sets.

    This function loads the Human Activity Segmentation challenge data sets into
    memory, downloading from GitHub
    (https://github.com/patrickzib/human_activity_segmentation_challenge) [1] if the
    data is not available at the specified ``extract_path``. The data sets were used
    in the discovery challenge held at ECML/PKDD and AALTD 2023. They contain 250
    annotated TS with 1-15 segments, capturing a total of 15 students performing 6
    distinct motion sequences. TS are sampled at 50 Hz, multivariate and consist of
    measurements from 9 out 12 smartphone sensors: triaxial accelerometer, gyroscope,
    magnetometer as well as latitude, longitude, and speed. Annotations include
    information about the challenge split (public / private), groups and subjects,
    as well as activity transition offsets (the change points) and activity labels.

    If you do not specify ``extract_path``, it will set the path to
    ``aeon/datasets/local_data``. If the problem is not present in ``extract_path``, it
    will attempt to download the data.

    Parameters
    ----------
    extract_path : str, default=None
        The path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/local_data/`. If a path is given, it can be an absolute,
        e.g., C:/Temp/ or relative, e.g. Temp/ or ./Temp/, path to an existing CSV-file.
    return_metadata : boolean, default = False
        If True, returns a tuple (X, y, metadata).

    Returns
    -------
    X: list of np.ndarray
        The list of multivariate (2d) time series with variable shape (n_instances, 9).
    y: list of np.ndarray
        The list of change points for every time series.
    metadata: optional
        The list of tuples containing data set names, splits, groups, subjects, and
        activities information.

    Raises
    ------
    URLError or HTTPError
        If the GitHub repository is not accessible.

    Examples
    --------
    >>> from aeon.datasets import load_human_activity_segmentation_datasets
    >>> X, y = load_human_activity_segmentation_datasets()
    ... )  # doctest: +SKIP

    References
    ----------
    .. [1] Arik Ermshaus, Patrick Schäfer, Anthony Bagnall, Thomas Guyet,
           Georgiana Ifrim, Vincent Lemaire, Ulf Leser, Colin Leverger,
           Simon Malinowski: Human Activity Segmentation Challenge @ ECML/PKDD’23.
           AALTD@ECML, 2023, DOI:10.1007/978-3-031-49896-1_1.
    """
    # set default/custom data folder
    if extract_path is not None:
        data_folder = Path(extract_path)
    else:
        data_folder = _DATA_FOLDER

    benchmark_path = _DATA_FOLDER / "has.csv"

    # converters to correctly load benchmark
    np_cols = [
        "change_points",
        "activities",
        "x-acc",
        "y-acc",
        "z-acc",
        "x-gyro",
        "y-gyro",
        "z-gyro",
        "x-mag",
        "y-mag",
        "z-mag",
        "lat",
        "lon",
        "speed",
    ]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val))
        for col in np_cols
    }

    # load activity data from git repo (and save locally) / or load locally
    if not benchmark_path.exists():
        data_folder.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(_HAS_URL, converters=converters, compression="zip")

        # make sure numerical data is correctly saved
        for np_col in np_cols:
            df[np_col] = df[np_col].apply(np.ndarray.tolist)
        df.to_csv(benchmark_path, index=None)

    df = pd.read_csv(benchmark_path, converters=converters)

    # construct return data
    X, y, metadata = list(), list(), list()

    for _, row in df.iterrows():
        dataset = (
            f"{row.group}_subject{row.subject}_routine{row.routine} "
            f"(id{row.ts_challenge_id})"
        )

        if row.group == "indoor":
            ts = np.hstack(
                (
                    row["x-acc"].reshape(-1, 1),
                    row["y-acc"].reshape(-1, 1),
                    row["z-acc"].reshape(-1, 1),
                    row["x-gyro"].reshape(-1, 1),
                    row["y-gyro"].reshape(-1, 1),
                    row["z-gyro"].reshape(-1, 1),
                    row["x-mag"].reshape(-1, 1),
                    row["y-mag"].reshape(-1, 1),
                    row["z-mag"].reshape(-1, 1),
                )
            )
        elif row.group == "outdoor":
            ts = np.hstack(
                (
                    row["x-acc"].reshape(-1, 1),
                    row["y-acc"].reshape(-1, 1),
                    row["z-acc"].reshape(-1, 1),
                    row["x-mag"].reshape(-1, 1),
                    row["y-mag"].reshape(-1, 1),
                    row["z-mag"].reshape(-1, 1),
                    row["lat"].reshape(-1, 1),
                    row["lon"].reshape(-1, 1),
                    row["speed"].reshape(-1, 1),
                )
            )

        X.append(ts)
        y.append(row.change_points)
        metadata.append((dataset, row.split, row.group, row.subject, row.activities))

    if return_metadata is True:
        return X, y, metadata

    return X, y
