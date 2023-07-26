# -*- coding: utf-8 -*-
"""Functions to load and collate results from timeseriesclassification.com."""
__all__ = [
    "get_estimator_results",
    "get_estimator_results_as_array",
    "get_available_estimators",
]
__author__ = ["TonyBagnall"]


import numpy as np
import pandas as pd

from aeon.datasets.tsc_data_lists import univariate as UCR

VALID_RESULT_TYPES = ["accuracy", "auroc", "balancedaccuracy", "nll"]
VALID_TASK_TYPES = ["classification", "clustering", "regression"]

NAME_ALIASES = {
    "Arsenal": {"ARSENAL", "TheArsenal", "AFC", "ArsenalClassifier"},
    "BOSS": {"TheBOSS", "boss", "BOSSClassifier"},
    "cBOSS": {"CBOSS", "CBOSSClassifier", "cboss"},
    "CIF": {"CanonicalIntervalForest", "CIFClassifier"},
    "CNN": {"cnn", "CNNClassifier"},
    "Catch22": {"catch22", "Catch22Classifier"},
    "DrCIF": {"DrCIF", "DrCIFClassifier"},
    "FreshPRINCE": {"FP", "freshPrince", "FreshPrince", "FreshPRINCEClassifier"},
    "HC1": {"HIVECOTE1", "HIVECOTEV1", "hc", "HIVE-COTEv1"},
    "HC2": {"HIVECOTE2", "HIVECOTEV2", "hc2", "HIVE-COTE", "HIVE-COTEv2"},
    "Hydra-MultiROCKET": {"Hydra-MR", "MultiROCKET-Hydra", "MR-Hydra", "HydraMR"},
    "InceptionTime": {"IT", "InceptionT", "inceptiontime", "InceptionTimeClassifier"},
    "MiniROCKET": {"MiniRocket", "MiniROCKETClassifier"},
    "MrSQM": {"mrsqm", "MrSQMClassifier"},
    "MultiROCKET": {"MultiRocket", "MultiROCKETClassifier"},
    "ProximityForest": {"PF", "ProximityForestV1", "PFV1"},
    "RDST": {"rdst", "RandomDilationShapeletTransform", "RDSTClassifier"},
    "RISE": {"RISEClassifier", "rise"},
    "ROCKET": {"Rocket", "RocketClassifier", "ROCKETClassifier"},
    "RSF": {"rsf", "RSFClassifier"},
    "RSTSF": {"R_RSTF", "RandomSTF", "RSTFClassifier"},
    "ResNet": {"R_RSTF", "RandomSTF", "RSTFClassifier"},
    "STC": {"ShapeletTransform", "STCClassifier", "RandomShapeletTransformClassifier"},
    "STSF": {"stsf", "STSFClassifier"},
    "Signatures": {"SignaturesClassifier"},
    "TDE": {"tde", "TDEClassifier"},
    "TS-CHIEF": {"TSCHIEF", "TS_CHIEF"},
    "TSF": {"tsf", "TimeSeriesForest"},
    "TSFresh": {"tsfresh", "TSFreshClassifier"},
    "WEASEL-Dilation": {"WEASEL", "WEASEL-D", "Weasel-D"},
}


def estimator_alias(name: str) -> str:
    """Return the standard name for possible aliased classifier.

    Parameters
    ----------
        name: str. Name of an estimator

    Returns
    -------
        str: standardised name as defined by NAME_ALIASES

    Example
    -------
    >>> from aeon.benchmarking.results_loaders import estimator_alias
    >>> estimator_alias("HIVECOTEV2")
    'HC2'
    """
    if name in NAME_ALIASES:
        return name
    for name_key in NAME_ALIASES.keys():
        if name in NAME_ALIASES[name_key]:
            return name_key
    raise ValueError(f"Unknown classifier name {name}")


def get_available_estimators(task="classification") -> pd.DataFrame:
    """Get a list of estimators avialable for a specific task.

    Parameters
    ----------
        task : str. default = "classification".
            this is not case sensitive. Should be one of
            "classification/clustering/regression

    Returns
    -------
        str: standardised name as defined by NAME_ALIASES

    Example
    -------
    >>> from aeon.benchmarking.results_loaders import get_available_estimators
    >>> cls = get_available_estimators("Classification")  #doctest: +SKIP
    """
    t = task.lower()
    if t not in VALID_TASK_TYPES:
        raise ValueError(
            f" task {t} is not available on tsc.com, must be one of {VALID_TASK_TYPES}"
        )
    path = (
        f"https://timeseriesclassification.com/results/ReferenceResults/"
        f"{t}/estimators.txt"
    )
    try:
        data = pd.read_csv(path)
    except Exception:
        raise ValueError(f"{path} is unavailable right now, try later")
    return data


def get_estimator_results(
    estimators: list,
    datasets=UCR,
    default_only=True,
    task="classification",
    type="accuracy",
    path="https://timeseriesclassification.com/results/ReferenceResults",
):
    """Look for results for given estimators for a list of datasets.

    This function pulls down a CSV of results, scans it for datasets and returns any
    results found. If a dataset is not present, it is ignored.

    Parameters
    ----------
    estimators: list of string.
        list of estimators to search for.
    datasets: list of string default = UCR.
        list of problem names to search for. Default is to look for the 112 UCR
        datasets listed in aeon.datasets.tsc_data_lists
    default_only: boolean, default = True
        whether to recover just the default test results, or 30 resamples
    path: string default https://timeseriesclassification.com/results/ReferenceResults/
        path where to read results from, default to tsc.com
    Returns
    -------
        results: list of dictionaries of dictionaries.
            list len(estimators) of dictionaries, each of which is a dictionary of
            dataset names for keys and results as the value. If default only is an
            np.ndarray.

    Example
    -------
    >>> from aeon.benchmarking.results_loaders import get_estimator_results
    >>> cls = ["HC2"]  # doctest: +SKIP
    >>> data = ["Chinatown", "Adiac"]  # doctest: +SKIP
    >>> get_estimator_results(estimators=cls, datasets=data) # doctest: +SKIP
    {'HC2': {'Chinatown': 0.9825072886297376, 'Adiac': 0.8107416879795396}}
    """
    task = task.lower()
    type = type.lower()
    if type not in VALID_RESULT_TYPES:
        raise ValueError(
            f"Error in get_estimator_results, {type} is not a valid type of " f"results"
        )

    if task not in VALID_TASK_TYPES:
        raise ValueError(f"Error in get_estimator_results, {task} is not a valid task")

    path = f"{path}/{task}/{type}/"
    suffix = "_TESTFOLDS.csv"
    all_results = {}
    for cls in estimators:
        alias_cls = estimator_alias(cls)
        url = path + alias_cls + suffix
        try:
            data = pd.read_csv(url)
        except Exception:
            raise ValueError(
                f"Cannot connect to {url} website down or results not " f"present"
            )
        cls_results = {}
        problems = data["folds:"]
        results = data.iloc[:, 1:].to_numpy()
        p = list(problems)
        for problem in datasets:
            if problem in p:
                pos = p.index(problem)
                if default_only:
                    cls_results[problem] = results[pos][0]
                else:
                    cls_results[problem] = results[pos]
        all_results[cls] = cls_results
    return all_results


def get_estimator_results_as_array(
    estimators: list,
    datasets=UCR,
    default_only=True,
    task="Classification",
    type="accuracy",
    include_missing=False,
    path="https://timeseriesclassification.com/results/ReferenceResults",
):
    """Look for results for given estimators for a list of datasets.

    This function pulls down a CSV of results, scans it for datasets and returns any
    results found. If a dataset is not present, it is ignored.

    Parameters
    ----------
    estimators: list of string.
        list of estimators to search for.
    datasets: list of string default = UCR.
        list of problem names to search for. Default is to look for the 112 UCR
        datasets listed in aeon.datasets.tsc_data_lists
    default_only: boolean, default = True
        whether to recover just the default test results, or 30 resamples. If false,
        values are averaged to get a 2D array.
    include_missing: boolean, default = False
        If a classifier does not have results for a given problem, either the whole
        problem is ignored when include_missing is False, or NaN
    path: string default https://timeseriesclassification.com/results/ReferenceResults/
        path where to read results from, default to tsc.com

    Returns
    -------
    results: 2D numpy array, each column is a results for a classifier, each row a
    dataset.
    if include_missing == false, returns names: an aligned list of names of included

    Example
    -------
    >>> from aeon.benchmarking.results_loaders import get_estimator_results
    >>> cls = ["HC2", "FreshPRINCE"] # doctest: +SKIP
    >>> data = ["Chinatown", "Adiac"] # doctest: +SKIP
    >>> get_estimator_results_as_array(estimators=cls, datasets=data) # doctest: +SKIP
    (array([[0.98250729, 0.98250729],
           [0.81074169, 0.84143223]]), ['Chinatown', 'Adiac'])
    """
    res_dicts = get_estimator_results(
        estimators=estimators,
        datasets=datasets,
        default_only=default_only,
        task=task,
        type=type,
        path=path,
    )
    all_res = []
    names = []
    for d in datasets:
        r = np.zeros(len(estimators))
        include = True
        for i in range(len(estimators)):
            temp = res_dicts[estimators[i]]
            if d in temp:
                if default_only:
                    r[i] = temp[d]
                else:
                    r[i] = np.average(temp[d])
            elif not include_missing:  # Skip whole problem
                include = False
                continue
            else:
                r[i] = False
        if include:
            all_res.append(r)
            names.append(d)

    if include_missing:
        return np.array(all_res)
    else:
        return np.array(all_res), names
