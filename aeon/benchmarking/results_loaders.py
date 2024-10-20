"""Functions to load and collate results from timeseriesclassification.com."""

__all__ = [
    "estimator_alias",
    "get_available_estimators",
    "get_estimator_results",
    "get_estimator_results_as_array",
]
__maintainer__ = []


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
    "Arsenal": {"ARSENAL", "TheArsenal", "AFC", "ArsenalClassifier"},
    "BOSS": {"TheBOSS", "boss", "BOSSClassifier", "BOSSEnsemble"},
    "cBOSS": {"CBOSS", "CBOSSClassifier", "cboss", "ContractableBOSS"},
    "CIF": {
        "CanonicalIntervalForest",
        "CIFClassifier",
        "CanonicalIntervalForestClassifier",
    },
    "CNN": {
        "cnn",
        "CNNClassifier",
        "CNNRegressor",
        "TimeCNNClassifier",
        "TimeCNNRegressor",
    },
    "Catch22": {"catch22", "Catch22Classifier"},
    "DrCIF": {"DrCIF", "DrCIFClassifier", "DrCIFRegressor"},
    "EE": {"ElasticEnsemble", "EEClassifier", "ElasticEnsembleClassifier"},
    "FreshPRINCE": {
        "FP",
        "freshPrince",
        "FreshPrince",
        "FreshPRINCEClassifier",
        "FreshPRINCERegressor",
    },
    "GRAIL": {"GRAILClassifier", "grail"},
    "HC1": {"HIVECOTE1", "HIVECOTEV1", "hc", "HIVE-COTEv1"},
    "HC2": {"HIVECOTE2", "HIVECOTEV2", "hc2", "HIVE-COTE", "HIVE-COTEv2"},
    "Hydra": {"hydra", "HydraClassifier"},
    "H-InceptionTime": {
        "H-IT",
        "H-InceptionT",
        "h-inceptiontime",
        "H-InceptionTimeClassifier",
    },
    "InceptionTime": {
        "IT",
        "InceptionT",
        "inceptiontime",
        "InceptionTimeClassifier",
        "InceptionTimeRegressor",
    },
    "LiteTime": {
        "LiteTimeClassifier",
        "litetime",
        "LITE",
        "LITETimeClassifier",
        "LITETime",
    },
    "MiniROCKET": {"MiniRocket", "MiniROCKETClassifier"},
    "MrSQM": {"mrsqm", "MrSQMClassifier"},
    "MR-Hydra": {
        "Hydra-MultiROCKET",
        "Hydra-MR",
        "MultiROCKET-Hydra",
        "HydraMR",
        "MultiRocketHydraClassifier",
        "MultiRocketHydra",
    },
    "MR": {
        "MultiRocket",
        "MultiROCKETClassifier",
        "MultiROCKETRegressor",
        "MultiROCKET",
    },
    "PF": {"ProximityForest", "ProximityForestV1", "PFV1"},
    "QUANT": {"quant", "QuantileForestClassifier", "QUANTClassifier"},
    "RDST": {"rdst", "RandomDilationShapeletTransform", "RDSTClassifier"},
    "RISE": {
        "RISEClassifier",
        "rise",
        "RandomIntervalSpectralEnsembleClassifier",
        "RandomIntervalSpectralEnsemble",
    },
    "RIST": {"RISTClassifier", "rist"},
    "ROCKET": {"Rocket", "RocketClassifier", "ROCKETClassifier", "ROCKETRegressor"},
    "RSF": {"rsf", "RSFClassifier"},
    "R-STSF": {"R_RSTF", "RandomSTF", "RSTFClassifier", "RSTSF"},
    "ResNet": {"resnet", "ResNetClassifier", "ResNetRegressor"},
    "STC": {
        "ShapeletTransform",
        "STCClassifier",
        "RandomShapeletTransformClassifier",
        "ShapeletTransformClassifier",
    },
    "STSF": {"stsf", "STSFClassifier", "SupervisedTimeSeriesForest"},
    "ShapeDTW": {"ShapeDTWClassifier"},
    "Signatures": {"SignaturesClassifier", "SignatureClassifier", "Signature"},
    "TDE": {"tde", "TDEClassifier", "TemporalDictionaryEnsemble"},
    "TS-CHIEF": {"TSCHIEF", "TS_CHIEF"},
    "TSF": {"tsf", "TimeSeriesForest", "TimeSeriesForestClassifier"},
    "TSFresh": {"tsfresh", "TSFreshClassifier"},
    "WEASEL-1.0": {"WEASEL", "WEASEL1", "weasel", "WEASEL 1.0"},
    "WEASEL-2.0": {"WEASEL-D", "WEASEL-Dilation", "WEASEL2", "weasel 2.0", "WEASEL_V2"},
    "1NN-DTW": {
        "1NNDTW",
        "1nn-dtw",
        "KNeighborsTimeSeriesRegressor",
        "KNeighborsTimeSeriesClassifier",
        "KNeighborsTimeSeries",
    },
    "1NN-ED": {
        "1NNED",
        "1nn-ed",
        "1nned",
    },
    "5NN-ED": {
        "5NNED",
        "5nn-ed",
        "5nned",
    },
    # Clustering
    "dtw-dba": {"DTW-DBA"},
    "kmeans-ed": {"ed-kmeans", "kmeans-euclidean", "k-means-ed", "KMeans-ED"},
    "kmeans-dtw": {"dtw-kmeans", "k-means-dtw", "KMeans-DTW"},
    "kmeans-msm": {"msm-kmeans", "k-means-msm", "KMeans-MSM"},
    "kmeans-twe": {"twe-kmeans", "k-means-twe", "KMeans-TWE"},
    "kmeans-ddtw": {"ddtw-kmeans"},
    "kmeans-edr": {"edr-kmeans"},
    "kmeans-erp": {"erp-kmeans"},
    "kmeans-lcss": {"lcss-kmeans"},
    "kmeans-wdtw": {"wdtw-kmeans"},
    "kmeans-wddtw": {"msm-kmeans"},
    "kmedoids-ed": {"ed-kmedoids", "k-medoids-ed", "KMedoids-ED"},
    "kmedoids-dtw": {"dtw-kmedoids", "k-medoids-dtw", "KMedoids-DTW"},
    "kmedoids-msm": {"msm-kmedoids", "k-medoids-msm", "KMedoids-MSM"},
    "kmedoids-twe": {"twe-kmedoids", "k-medoids-twe", "KMedoids-TWE"},
    "kmedoids-ddtw": {"ddtw-kmeans"},
    "kmedoids-edr": {"edr-kmedoids"},
    "kmedoids-erp": {"erp-kmedoids"},
    "kmedoids-lcss": {"lcss-kmedoids"},
    "kmedoids-wdtw": {"wdtw-kmedoids"},
    "kmedoids-wddtw": {"msm-kmedoids"},
    # Regression only
    "FCN": {"fcn", "FCNRegressor"},
    "FPCR": {"fpcr", "FPCRRegressor"},
    "FPCR-b-spline": {"fpcr-b-spline", "FPCRBSplineRegressor"},
    "GridSVR": {"gridSVR", "GridSVRRegressor"},
    "RandF": {"randf", "RandFRegressor"},
    "RotF": {"rotf", "RotFRegressor"},
    "Ridge": {"ridge", "RidgeRegressor"},
    "SingleInceptionTime": {"SIT", "SingleInceptionT", "SingleInceptionTimeRegressor"},
    "XGBoost": {"xgboost", "XGBoostRegressor"},
    "5NN-DTW": {"5NNDTW", "5nn-dtw"},
}

CONNECTION_ERRORS = [
    HTTPError,
    URLError,
    RemoteDisconnected,
    IncompleteRead,
    ConnectionResetError,
    TimeoutError,
]


def estimator_alias(name: str) -> str:
    """Return the standard name for possible aliased estimator.

    Parameters
    ----------
    name: str
        Name of an estimator.

    Returns
    -------
    name: str
        Standardized name as defined by NAME_ALIASES.

    Examples
    --------
    >>> from aeon.benchmarking.results_loaders import estimator_alias
    >>> estimator_alias("HIVECOTEV2")
    'HC2'
    """
    if name in NAME_ALIASES:
        return name
    for name_key in NAME_ALIASES.keys():
        if name in NAME_ALIASES[name_key]:
            return name_key
    raise ValueError(
        f"Unknown estimator name {name}. For a list of valid names and allowed "
        "aliases, see NAME_ALIASES in aeon/benchmarking/results_loaders.py. Note "
        "that estimator names are case sensitive."
    )


def get_available_estimators(task="classification", return_dataframe=True):
    """Get a list of estimators avialable for a specific task.

    Parameters
    ----------
    task : str, default="classification"
        Should be one of "classification","clustering","regression". This is not case
        sensitive.
    return_dataframe : boolean, default = True
        If false, returns a list.

    Returns
    -------
    pd.DataFrame or List
        Standardised name as defined by NAME_ALIASES.

    Examples
    --------
    >>> from aeon.benchmarking.results_loaders import get_available_estimators
    >>> cls = get_available_estimators("Classification")  # doctest: +SKIP
    """
    t = task.lower()
    if t not in VALID_TASK_TYPES:
        raise ValueError(
            f" task {t} is not available on tsc.com, must be one of {VALID_TASK_TYPES}"
        )
    path = (
        f"http://timeseriesclassification.com/results/ReferenceResults/"
        f"{t}/estimators.txt"
    )
    data = pd.read_csv(path)
    if return_dataframe:
        return data
    else:
        return data.iloc[:, 0].tolist()


def get_estimator_results(
    estimators: Union[str, list[str]],
    datasets: Optional[list[str]] = None,
    num_resamples: Optional[int] = 1,
    task: str = "classification",
    measure: str = "accuracy",
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

    probs_names = "Resamples:"
    path = f"{path}/{task}/{measure}/"
    results = {}

    for cls in estimators:
        url = path + estimator_alias(cls) + "_" + measure + ".csv"
        data = pd.read_csv(url)
        problems = list(data[probs_names].str.replace(r"_.*", "", regex=True))
        dsets = problems if datasets is None else datasets
        res_arr = data.iloc[:, 1:].to_numpy()

        cls_results = {}
        for data in dsets:
            if data in problems:
                pos = problems.index(data)
                if num_resamples == 1:
                    cls_results[data] = res_arr[pos][0]
                elif num_resamples is None:
                    cls_results[data] = res_arr[pos]
                else:
                    cls_results[data] = res_arr[pos][:num_resamples]

        results[cls] = cls_results

    return results


def get_estimator_results_as_array(
    estimators: Union[str, list[str]],
    datasets: Optional[list[str]] = None,
    num_resamples: Optional[int] = 1,
    task: str = "classification",
    measure: str = "accuracy",
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
        path=path,
    )

    if datasets is None:
        datasets = []
        for cls in res_dict:
            datasets.extend(res_dict[cls].keys())
        datasets = set(datasets)

    results = []
    names = []
    for data in datasets:
        r = np.zeros(len(estimators))
        include = True
        for i in range(len(estimators)):
            if data in res_dict[estimators[i]]:
                r[i] = np.average(res_dict[estimators[i]][data])
            elif not include_missing:  # Skip the whole problem
                include = False
                break
            else:
                r[i] = np.NaN
        if include:
            results.append(r)
            names.append(data)

    return np.array(results), names
