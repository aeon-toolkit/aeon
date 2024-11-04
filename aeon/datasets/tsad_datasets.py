"""Datasets in the TimeEval data archive.

TimeEval provides time series anomaly detection datasets from different sources and is
hosted at https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html. The data
is actually stored on HiDrive and can be downloaded from there.

TimeEval provides its own dataset format, documented at
https://timeeval.readthedocs.io/en/latest/concepts/datasets.html#canonical-file-format.
Datasets are organized into dataset collections that bundle time series from the same
source or domain. Each dataset comes with a test time series in a CSV file and
potentially an additional training time series in a second file. The anomalies are
labeled per time point using a binary label (0 = normal, 1 = anomalous) with the
header "is_anomaly" within the time series files. There are roughly 11.5k time series
grouped in 30 collections in the TimeEval archive.

All time series have no missing values and assume equidistant time points. TimeEval
distinguishes between three learning types and two time series types.

Learning types:

- unsupervised: An unsupervised dataset consists only of a single test time series.
- supervised: A supervised dataset consists of a training time series with anomalies
  and a test time series.
- semi-supervised: A semi-supervised dataset consists of a training time series with
  normal data (no anomalies; all labels are 0) and a test time series.

Time series types:

- univariate: Univariate datasets consist of a single feature/dimension/channel.
- multivariate: Multivariate datasets have two or more features/dimensions/channels.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import aeon

_DATA_FOLDER = Path(aeon.__file__).parent / "datasets" / "local_data"
_TIMEEVAL_INDEX_URL = "https://my.hidrive.com/api/sharelink/download?id=koCGAvOe"


def _to_tuple_list(X: np.ndarray) -> list[tuple[str, str]]:
    return [tuple(x) for x in X]


def _load_indexfile(refresh=False) -> pd.DataFrame:
    """Load dataset and collection information from the TimeEval index file."""
    if refresh or not (_DATA_FOLDER / "datasets.csv").exists():
        _DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(_TIMEEVAL_INDEX_URL)
        df.to_csv(_DATA_FOLDER / "_datasets.csv", index=False)
        # Atomic (unfortunately only guaranteed on POSIX-systems) rename to avoid
        # reading incomplete file in other processes. Atomicity guarantees are hard to
        # find in Windows (NT).
        os.rename(_DATA_FOLDER / "_datasets.csv", _DATA_FOLDER / "datasets.csv")
    else:
        df = pd.read_csv(_DATA_FOLDER / "datasets.csv")

    return df


def tsad_collections() -> dict[str, list[str]]:
    """Return dictionary mapping collection names to dataset names."""
    df = _load_indexfile()
    return (
        df.groupby("collection_name")
        .apply(lambda x: x["dataset_name"].to_list(), include_groups=False)
        .to_dict()
    )


def tsad_datasets() -> list[tuple[str, str]]:
    """Return list of all anomaly detection datasets in the TimeEval archive."""
    df = _load_indexfile()
    return _to_tuple_list(df[["collection_name", "dataset_name"]].values)


def univariate() -> list[tuple[str, str]]:
    """Return list of univariate anomaly detection datasets."""
    df = _load_indexfile()
    return _to_tuple_list(
        df.loc[
            df["input_type"] == "univariate", ["collection_name", "dataset_name"]
        ].values
    )


def multivariate() -> list[tuple[str, str]]:
    """Return list of multivariate anomaly detection datasets."""
    df = _load_indexfile()
    return _to_tuple_list(
        df.loc[
            df["input_type"] == "multivariate", ["collection_name", "dataset_name"]
        ].values
    )


def unsupervised() -> list[tuple[str, str]]:
    """Return list of unsupervised anomaly detection datasets."""
    df = _load_indexfile()
    return _to_tuple_list(
        df.loc[
            df["train_type"] == "unsupervised", ["collection_name", "dataset_name"]
        ].values
    )


def supervised() -> list[tuple[str, str]]:
    """Return list of supervised anomaly detection datasets."""
    df = _load_indexfile()
    return _to_tuple_list(
        df.loc[
            df["train_type"] == "supervised", ["collection_name", "dataset_name"]
        ].values
    )


def semi_supervised() -> list[tuple[str, str]]:
    """Return list of semi-supervised anomaly detection datasets."""
    df = _load_indexfile()
    return _to_tuple_list(
        df.loc[
            df["train_type"] == "semi-supervised", ["collection_name", "dataset_name"]
        ].values
    )
