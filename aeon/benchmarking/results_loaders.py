# -*- coding: utf-8 -*-
"""Functions to load and collate results from timeseriesclassification.com."""
__all__ = ["get_results_from_tsc_com", "get_array_from_tsc_com"]

import numpy as np
import pandas as pd

from aeon.datasets.tsc_dataset_names import univariate as UCR

VALID_RESULT_TYPES = ["Accuracy", "AUROC", "BalancedAccuracy"]
VALID_TASK_TYPES = ["Classification", "Clustering", "Regression"]


def get_avialable_estimators(task="Classification"):
    """Get a list of classifiers avialable for a specific task."""
    path = (
        f"https://timeseriesclassification.com/results/ReferenceResults/"
        f"{task}/estimators.txt"
    )
    data = pd.read_csv(path)
    return data


def get_results_from_tsc_com(
    classifiers: list,
    datasets=UCR,
    default_only=True,
    task="Classification",
    type="Accuracy",
):
    """Look for results for given classifiers for a list of datasets.

    This function pulls down a CSV of results, scans it for datasets and returns any
    results found. If a dataset is not present, it is ignored.

    Parameters
    ----------
    classifiers: list of string.
        list of classifiers to search for.
    datasets: list of string default = UCR.
        list of problem names to search for. Default is to look for the 112 UCR
        datasets listed in aeon.datasets.tsc_dataset_names
    default_only: boolean, default = True
        whether to recover just the default test results, or 30 resamples
    Returns
    -------
        results: list of dictionaries of dictionaries.
            list len(classifiers) of dictionaries, each of which is a dictionary of
            dataset names for keys and results as the value. If default only is an
            np.ndarray.

    Example
    -------
    >>> from aeon.benchmarking.results_loaders import get_results_from_tsc_com
    >>> cls = ["HC2"]
    >>> data = ["Chinatown", "Adiac"]
    >>> get_results_from_tsc_com(classifiers=cls, datasets=data)
    [{'HC2': {'Chinatown': 0.9825072886297376, 'Adiac': 0.8107416879795396}}]
    """
    if type not in VALID_RESULT_TYPES:
        raise ValueError(
            f"Error in get_results_from_tsc_com, {type} is not a valid type of "
            f"results"
        )
    if task not in VALID_TASK_TYPES:
        raise ValueError(
            f"Error in get_results_from_tsc_com, {task} is not a valid task"
        )

    path = (
        f"https://timeseriesclassification.com/results/ReferenceResults/{task}/"
        f"{type}/"
    )
    suffix = "_TESTFOLDS.csv"
    all_results = {}
    for cls in classifiers:
        url = path + cls + suffix
        data = pd.read_csv(url)
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


def get_array_from_tsc_com(
    classifiers: list,
    datasets=UCR,
    default_only=True,
    task="Classification",
    type="Accuracy",
    include_missing=False,
):
    """Look for results for given classifiers for a list of datasets.

    This function pulls down a CSV of results, scans it for datasets and returns any
    results found. If a dataset is not present, it is ignored.

    Parameters
    ----------
    classifiers: list of string.
        list of classifiers to search for.
    datasets: list of string default = UCR.
        list of problem names to search for. Default is to look for the 112 UCR
        datasets listed in aeon.datasets.tsc_dataset_names
    default_only: boolean, default = True
        whether to recover just the default test results, or 30 resamples. If false,
        values are averaged to get a 2D array.
    include_missing: boolean, default = False
        If a classifier does not have results for a given problem, either the whole
        problem is ignored when include_missing is False, or NaN

    Returns
    -------
    results: 2D numpy array, each column is a results for a classifier, each row a
    dataset.
    if include_missing == false, returns names: an aligned list of names of included
    """
    res_dicts = get_results_from_tsc_com(
        classifiers=classifiers,
        datasets=datasets,
        default_only=default_only,
        task=task,
        type=type,
    )
    all_res = []
    names = []
    for d in datasets:
        r = np.zeros(len(classifiers))
        include = True
        for i in range(len(classifiers)):
            temp = res_dicts[classifiers[i]]
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
