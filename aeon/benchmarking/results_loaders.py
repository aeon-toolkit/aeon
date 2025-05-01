"""Functions to load and collate results from timeseriesclassification.com."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "estimator_alias",
    "get_available_estimators",
    "get_estimator_results",
    "get_estimator_results_as_array",
]


from http.client import IncompleteRead, RemoteDisconnected
from typing import Optional, Union
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

VALID_TASK_TYPES = ["classification", "clustering", "regression"]

VALID_RESULT_MEASURES = {
    "classification": ["accuracy", "auroc", "balacc", "f1", "logloss"],
    "clustering": ["clacc", "ami", "ari", "mi"],
    "regression": ["mse", "mae", "r2", "mape", "rmse"],
}

NAME_ALIASES = {
    # convolution based
    "Arsenal": ["TheArsenal", "AFC", "ArsenalClassifier"],
    "ROCKET": ["ROCKETClassifier", "ROCKETRegressor"],
    "MiniROCKET": ["MiniROCKETClassifier"],
    "MR": ["MultiROCKET", "MultiROCKETClassifier"],
    "Hydra": ["hydraclassifier"],
    "MR-Hydra": [
        "Hydra-MultiROCKET",
        "Hydra-MR",
        "MultiROCKET-Hydra",
        "HydraMR",
        "MultiRocketHydraClassifier",
        "MultiRocketHydra",
    ],
    # deep learning
    "CNN": [
        "CNNClassifier",
        "CNNRegressor",
        "TimeCNNClassifier",
        "TimeCNNRegressor",
    ],
    "FCN": ["FCNRegressor"],
    "ResNet": ["ResNetClassifier", "ResNetRegressor"],
    "SingleInceptionTime": ["SIT", "SingleInceptionT", "SingleInceptionTimeRegressor"],
    "InceptionTime": [
        "IT",
        "InceptionT",
        "InceptionTimeClassifier",
        "InceptionTimeRegressor",
    ],
    "H-InceptionTime": ["H-IT", "H-InceptionT", "H-InceptionTimeClassifier"],
    "LiteTime": ["LITE", "LITETimeClassifier"],
    # dictionary based
    "BOSS": ["theboss", "bossclassifier", "bossensemble"],
    "cBOSS": ["CBOSSClassifier", "ContractableBOSS"],
    "TDE": ["TDEClassifier", "TemporalDictionaryEnsemble"],
    "WEASEL-1.0": ["WEASEL", "WEASEL1", "WEASEL 1.0"],
    "WEASEL-2.0": [
        "WEASEL-D",
        "WEASEL-Dilation",
        "WEASEL2",
        "WEASEL 2.0",
        "WEASEL_V2",
        "W 2.0",
    ],
    "MrSQM": ["MrSQMClassifier"],
    # distance based
    "1NN-DTW": [
        "1NNDTW",
        "KNeighborsTimeSeriesRegressor",
        "KNeighborsTimeSeriesClassifier",
        "KNeighborsTimeSeries",
    ],
    "5NN-DTW": ["5NNDTW"],
    "1NN-ED": ["1NNED"],
    "5NN-ED": ["5NNED"],
    "ShapeDTW": ["ShapeDTWClassifier"],  # bad results?
    "GRAIL": ["GRAILClassifier"],
    "EE": ["ElasticEnsemble", "EEClassifier", "ElasticEnsembleClassifier"],
    "PF": ["ProximityForest", "ProximityForestV1", "PFV1"],
    # feature based
    "Catch22": ["Catch22Classifier"],
    "FreshPRINCE": [
        "FP",
        "FreshPRINCEClassifier",
        "FreshPRINCERegressor",
    ],
    "Signatures": ["SignaturesClassifier", "SignatureClassifier", "Signature"],
    "TSFresh": ["TSFreshClassifier"],
    "FPCR": ["FPCRRegressor"],
    "FPCR-b-spline": ["FPCRBSplineRegressor"],
    # hybrid
    "HC1": ["hivecote1", "hivecotev1", "hive-cotev1"],
    "HC2": ["hivecote2", "hivecotev2", "hive-cote", "hive-cotev2"],
    "TS-CHIEF": ["TSCHIEF", "TS_CHIEF"],
    "RIST": ["RISTClassifier"],
    # interval based
    "TSF": ["TimeSeriesForest", "TimeSeriesForestClassifier"],
    "RISE": [
        "RISEClassifier",
        "RandomIntervalSpectralEnsembleClassifier",
        "RandomIntervalSpectralEnsemble",
    ],
    "CIF": [
        "CanonicalIntervalForest",
        "CIFClassifier",
        "CanonicalIntervalForestClassifier",
    ],
    "DrCIF": ["DrCIFClassifier", "DrCIFRegressor"],
    "STSF": ["STSFClassifier", "SupervisedTimeSeriesForest"],
    "R-STSF": ["R_RSTF", "RandomSTF", "RSTFClassifier", "RSTSF"],
    "QUANT": ["QuantileForestClassifier", "QUANTClassifier"],
    # shapelet based
    "STC": [
        "ShapeletTransform",
        "STCClassifier",
        "RandomShapeletTransformClassifier",
        "ShapeletTransformClassifier",
    ],
    "RSF": ["RSFClassifier"],
    "RDST": ["RandomDilationShapeletTransform", "RDSTClassifier"],
    # distance clustering
    "dtw-dba": [],
    "kmeans-ed": ["ed-kmeans", "kmeans-euclidean", "k-means-ed"],
    "kmeans-dtw": ["dtw-kmeans", "k-means-dtw"],
    "kmeans-msm": ["msm-kmeans", "k-means-msm"],
    "kmeans-twe": ["twe-kmeans", "k-means-twe"],
    "kmeans-ddtw": ["ddtw-kmeans"],
    "kmeans-edr": ["edr-kmeans"],
    "kmeans-erp": ["erp-kmeans"],
    "kmeans-lcss": ["lcss-kmeans"],
    "kmeans-wdtw": ["wdtw-kmeans"],
    "kmeans-wddtw": ["msm-kmeans"],
    "kmedoids-ed": ["ed-kmedoids", "k-medoids-ed"],
    "kmedoids-dtw": ["dtw-kmedoids", "k-medoids-dtw"],
    "kmedoids-msm": ["msm-kmedoids", "k-medoids-msm"],
    "kmedoids-twe": ["twe-kmedoids", "k-medoids-twe"],
    "kmedoids-ddtw": ["ddtw-kmeans"],
    "kmedoids-edr": ["edr-kmedoids"],
    "kmedoids-erp": ["erp-kmedoids"],
    "kmedoids-lcss": ["lcss-kmedoids"],
    "kmedoids-wdtw": ["wdtw-kmedoids"],
    "kmedoids-wddtw": ["msm-kmedoids"],
    # vector classifiers
    "GridSVR": ["GridSVRRegressor"],
    "RandF": ["RandFRegressor"],
    "RotF": ["RotFRegressor"],
    "Ridge": ["RidgeRegressor"],
    "XGBoost": ["XGBoostRegressor"],
}

CONNECTION_ERRORS = (
    HTTPError,
    URLError,
    RemoteDisconnected,
    IncompleteRead,
    ConnectionResetError,
    TimeoutError,
)


def estimator_alias(name: str) -> str:
    """Return the standard name for possible aliased estimator.

    Parameters
    ----------
    name: str
        Name of an estimator. Not case-sensitive.

    Returns
    -------
    name: str
        Standardised name as defined by NAME_ALIASES.

    Examples
    --------
    >>> from aeon.benchmarking.results_loaders import estimator_alias
    >>> estimator_alias("HIVECOTEV2")
    'HC2'
    """
    nl = name.lower()
    for name_key in NAME_ALIASES.keys():
        if nl == name_key.lower():
            return name_key
        for alias in NAME_ALIASES[name_key]:
            if nl == alias.lower():
                return name_key
    raise ValueError(
        f"Unknown estimator name {name}. For a list of valid names and allowed "
        "aliases, see NAME_ALIASES in aeon/benchmarking/results_loaders.py."
    )


def get_available_estimators(
    task: str = "classification", as_list: bool = False
) -> Union[pd.DataFrame, list]:
    """Get a DataFrame of estimators avialable for a specific learning task.

    Parameters
    ----------
    task: str, default="classification"
        A learning task contained within VALID_TASK_TYPES i.e. "classification",
        "clustering", "regression". Not case-sensitive.
    as_list: boolean, default=False
        If True, returns a list instead of a dataframe.

    Returns
    -------
    data: pd.DataFrame or list
        Standardised name as defined by NAME_ALIASES.

    Examples
    --------
    >>> from aeon.benchmarking.results_loaders import get_available_estimators
    >>> cls = get_available_estimators("Classification")  # doctest: +SKIP
    """
    t = task.lower()
    if t not in VALID_TASK_TYPES:
        raise ValueError(
            f"Learning task {t} is not available on timeseriesclassification.com, must "
            f"be one of {VALID_TASK_TYPES}"
        )
    data = pd.read_csv(
        f"http://timeseriesclassification.com/results/ReferenceResults/"
        f"{t}/estimators.txt"
    )
    return data.iloc[:, 0].tolist() if as_list else data


def get_estimator_results(
    estimators: Union[str, list[str]],
    datasets: Optional[list[str]] = None,
    num_resamples: Optional[int] = 1,
    task: str = "classification",
    measure: str = "accuracy",
    remove_dataset_modifiers: bool = False,
    path: str = "http://timeseriesclassification.com/results/ReferenceResults",
):
    """Look for results for given estimators for a list of datasets.

    This function loads or pulls down a CSV of results, scans it for datasets and
    returns any results found as a dictionary. If a dataset is not present, it is
    ignored.

    Parameters
    ----------
    estimators : str ot list of str
        Estimator name or list of estimator names to search for. See
        get_available_estimators, aeon.benchmarking.results_loading.NAME_ALIASES or
        the directory at path for valid options.
    datasets : list of str or None, default=None
        List of problem names to search for. If the dataset is not present in the
        results, it is ignored.
        If None, all datasets the estimator has results for is returned.
    num_resamples : int or None, default=1
        The number of data resamples to return scores for. The first resample
        is the default train/test split for the dataset.
        For 1, only the score for the default train/test split of the dataset is
        returned.
        For 2 or more, a np.ndarray of scores for all resamples up to num_resamples are
        returned.
        If None, the scores of all resamples are returned.
    task : str, default="classification"
        Should be one of aeon.benchmarking.results_loading.VALID_TASK_TYPES. i.e.
        "classification", "clustering", "regression".
    measure : str, default="accuracy"
        Should be one of aeon.benchmarking.results_loading.VALID_RESULT_MEASURES[task].
        Dependent on the task, i.e. for classification, "accuracy", "auroc", "balacc",
        and regression, "mse", "mae", "r2".
    remove_dataset_modifiers: bool, default=False
        If True, will remove any dataset modifier (anything after the first underscore)
        from the dataset names in the loaded results file.
        i.e. a loaded result row for "Dataset_eq" will be converted to just "Dataset".
    path : str, default="https://timeseriesclassification.com/results/ReferenceResults/"
        Path where to read results from. Defaults to timeseriesclassification.com.

    Returns
    -------
    results: dict
        Dictionary with estimator name keys containing another dictionary.
        Sub-dictionary consists of dataset name keys and contains of scores for each
        dataset.

    Examples
    --------
    >>> from aeon.benchmarking.results_loaders import get_estimator_results
    >>> cls = ["HC2"]  # doctest: +SKIP
    >>> data = ["Chinatown", "Adiac"]  # doctest: +SKIP
    >>> get_estimator_results(estimators=cls, datasets=data) # doctest: +SKIP
    {'HC2': {'Chinatown': 0.9825072886297376, 'Adiac': 0.8107416879795396}}
    """
    task = task.lower()
    measure = measure.lower()
    if task not in VALID_TASK_TYPES:
        raise ValueError(f"Error in get_estimator_results, {task} is not a valid task")
    if measure not in VALID_RESULT_MEASURES[task]:
        raise ValueError(
            f"Error in get_estimator_results, {measure} is not a valid type of "
            f"results for task {task}"
        )
    if not isinstance(estimators, list):
        estimators = [estimators]
    path = f"{path}/{task}/{measure}/"

    return _load_to_dict(
        path=path,
        estimators=estimators,
        datasets=datasets,
        num_resamples=num_resamples,
        file_suffix=f"_{measure}.csv",
        est_alias=True,
        remove_data_modifier=remove_dataset_modifiers,
    )


def get_estimator_results_as_array(
    estimators: Union[str, list[str]],
    datasets: Optional[list[str]] = None,
    num_resamples: Optional[int] = 1,
    task: str = "classification",
    measure: str = "accuracy",
    remove_dataset_modifiers: bool = False,
    path: str = "http://timeseriesclassification.com/results/ReferenceResults",
    include_missing: bool = False,
):
    """Look for results for given estimators for a list of datasets.

    This function loads or pulls down a CSV of results, scans it for datasets and
    returns any results found as an array. If a dataset is not present, it is ignored.

    Parameters
    ----------
    estimators : list of str
        Estimator name or list of estimator names to search for. See
        get_available_estimators, aeon.benchmarking.results_loading.NAME_ALIASES or
        the directory at path for valid options.
    datasets : list of or None, default=1
        List of problem names to search for.
        If None, all datasets the estimator has results for is returned.
        If the dataset is not present in any of the results, it is ignored unless
        include_missing is true.
    num_resamples : int or None, default=None
        The number of data resamples to average over for all scores. The first resample
        is the default train/test split for the dataset.
        For 1, only the score for the default train/test split of the dataset is
        returned.
        For 2 or more, the scores of all resamples up to num_resamples are averaged and
        returned.
        If None, the scores of all resamples are averaged and returned.
    task : str, default="classification"
        Should be one of aeon.benchmarking.results_loading.VALID_TASK_TYPES. i.e.
        "classification", "clustering", "regression".
    measure : str, default="accuracy"
        Should be one of aeon.benchmarking.results_loading.VALID_RESULT_MEASURES[task].
        Dependent on the task, i.e. for classification, "accuracy", "auroc", "balacc",
        and regression, "mse", "mae", "r2".
    remove_dataset_modifiers: bool, default=False
        If True, will remove any dataset modifier (anything after the first underscore)
        from the dataset names in the loaded results file.
        i.e. a loaded result row for "Dataset_eq" will be converted to just "Dataset".
    path : str, default="https://timeseriesclassification.com/results/ReferenceResults/"
        Path where to read results from. Defaults to timeseriesclassification.com.
    include_missing : bool, default=False
        Whether to include datasets with missing results in the output.
        If False, the whole problem is ignored if any estimator is missing results it.
        If True, NaN is returned instead of a score in missing cases.

    Returns
    -------
    results: 2D numpy array
        Array of scores. Each column is a results for a classifier, each row a dataset.
    names: list of str
        List of dataset names that were retained.

    Examples
    --------
    >>> from aeon.benchmarking.results_loaders import get_estimator_results
    >>> cls = ["HC2", "FreshPRINCE"] # doctest: +SKIP
    >>> data = ["Chinatown", "Adiac"] # doctest: +SKIP
    >>> get_estimator_results_as_array(estimators=cls, datasets=data) # doctest: +SKIP
    (array([[0.98250729, 0.98250729],
           [0.81074169, 0.84143223]]), ['Chinatown', 'Adiac'])
    """
    if not isinstance(estimators, list):
        estimators = [estimators]

    res_dict = get_estimator_results(
        estimators=estimators,
        datasets=datasets,
        num_resamples=num_resamples,
        task=task,
        measure=measure,
        remove_dataset_modifiers=remove_dataset_modifiers,
        path=path,
    )

    if datasets is None:
        datasets = []
        for cls in res_dict:
            datasets.extend(res_dict[cls].keys())
        datasets = set(datasets)

    return _results_dict_to_array(res_dict, estimators, datasets, include_missing)


def _load_to_dict(
    path,
    estimators,
    datasets,
    num_resamples,
    file_suffix,
    est_alias=True,
    remove_data_modifier=False,
    csv_header="infer",
    ignore_nan=False,
):
    results = {}
    for est in estimators:
        est_name = estimator_alias(est) if est_alias else est
        url = path + est_name + file_suffix
        data = pd.read_csv(url, header=csv_header)
        problems = (
            list(data.iloc[:, 0].str.replace(r"_.*", "", regex=True))
            if remove_data_modifier
            else list(data.iloc[:, 0])
        )
        dsets = problems if datasets is None else datasets
        res_arr = data.iloc[:, 1:].to_numpy()

        est_results = {}
        for data in dsets:
            if data in problems:
                pos = problems.index(data)
                if num_resamples == 1:
                    est_results[data] = res_arr[pos][0]
                elif num_resamples is None:
                    est_results[data] = res_arr[pos]
                else:
                    est_results[data] = res_arr[pos][:num_resamples]

                if not ignore_nan and np.isnan(est_results[data]).any():
                    raise ValueError(
                        f"Missing resamples for {data} in {est}: {est_results[data]}"
                    )

        results[est] = est_results
    return results


def _results_dict_to_array(res_dict, estimators, datasets, include_missing):
    results = []
    names = []
    for data in datasets:
        r = np.zeros(len(estimators))
        include = True
        for i in range(len(estimators)):
            if data in res_dict[estimators[i]]:
                r[i] = np.nanmean(res_dict[estimators[i]][data])
            elif not include_missing:  # Skip the whole problem
                include = False
                break
            else:
                r[i] = np.nan
        if include:
            results.append(r)
            names.append(data)
    return np.array(results), names
