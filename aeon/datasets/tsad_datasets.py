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


def _load_from_indexfile(refresh=False) -> tuple:
    """Load dataset and collection information from the TimeEval index file."""
    from pathlib import Path

    import pandas as pd

    import aeon

    _DATA_FOLDER = Path(aeon.__file__).parent / "datasets" / "tsad_data"
    _TIMEEVAL_INDEX_URL = "https://my.hidrive.com/api/sharelink/download?id=koCGAvOe"

    def _to_tuple_list(X):
        return [tuple(x) for x in X]

    if refresh:
        df = pd.read_csv(_TIMEEVAL_INDEX_URL)
        df.to_csv(_DATA_FOLDER / "datasets.csv", index=False)
    else:
        df = pd.read_csv(_DATA_FOLDER / "datasets.csv")

    _tsad_collections = (
        df.groupby("collection_name")
        .apply(lambda x: x["dataset_name"].to_list())
        .to_dict()
    )
    _tsad_datasets = _to_tuple_list(df[["collection_name", "dataset_name"]].values)
    _univariate = _to_tuple_list(
        df.loc[
            df["input_type"] == "univariate", ["collection_name", "dataset_name"]
        ].values
    )
    _multivariate = _to_tuple_list(
        df.loc[
            df["input_type"] == "multivariate", ["collection_name", "dataset_name"]
        ].values
    )
    _unsupervised = _to_tuple_list(
        df.loc[
            df["train_type"] == "unsupervised", ["collection_name", "dataset_name"]
        ].values
    )
    _supervised = _to_tuple_list(
        df.loc[
            df["train_type"] == "supervised", ["collection_name", "dataset_name"]
        ].values
    )
    _semi_supervised = _to_tuple_list(
        df.loc[
            df["train_type"] == "semi-supervised", ["collection_name", "dataset_name"]
        ].values
    )
    return (
        _tsad_collections,
        _tsad_datasets,
        _univariate,
        _multivariate,
        _unsupervised,
        _supervised,
        _semi_supervised,
    )


(
    tsad_collections,
    tsad_datasets,
    univariate,
    multivariate,
    unsupervised,
    supervised,
    semi_supervised,
) = _load_from_indexfile()
