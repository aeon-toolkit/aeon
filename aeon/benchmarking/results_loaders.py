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
    "WEASEL-Dilation": {"WEASEL", "WEASEL-D", "Weasel-D", "WEASEL2"},
    "kmeans-ed": {"ed-kmeans", "kmeans-euclidean", "k-means-ed"},
    "kmeans-dtw": {"dtw-kmeans", "k-means-dtw"},
    "kmeans-msm": {"msm-kmeans", "k-means-msm"},
    "kmeans-twe": {"msm-kmeans", "k-means-msm"},
    "kmedoids-ed": {"ed-kmedoids", "k-medoids-ed"},
    "kmedoids-dtw": {"dtw-kmedoids", "k-medoids-dtw"},
    "kmedoids-msm": {"msm-kmedoids", "k-medoids-msm"},
    "kmedoids-twe": {"twe-kmedoids", "k-medoids-twe"},
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
    raise ValueError(f"Unknown estimator name {name}")


def get_available_estimators(task="classification") -> pd.DataFrame:
    """Get a list of estimators avialable for a specific task.

    Parameters
    ----------
    task : str, default="classification"
        Should be one of "classification","clustering","regression". This is not case
        sensitive.

    Returns
    -------
    str
        Standardised name as defined by NAME_ALIASES.

    Example
    -------
    >>> from aeon.benchmarking.results_loaders import get_available_estimators
    >>> cls = get_available_estimators("Classification")  # doctest: +SKIP
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
    suffix="_TESTFOLDS.csv",
):
    """Look for results for given estimators for a list of datasets.

    This function loads or pulls down a CSV of results, scans it for datasets and
    returns any results found. If a dataset is not present, it is ignored.

    Parameters
    ----------
    estimators : list of str
        list of estimators to search for.
    datasets : list of str, default = UCR
        list of problem names to search for. Default is to look for the 112 UCR
        datasets listed in aeon.datasets.tsc_data_lists.
    default_only : boolean, default = True
        Whether to recover just the default test results, or 30 resamples.
    path : str, default="https://timeseriesclassification.com/results/ReferenceResults/"
        Path where to read results from, default to tsc.com
    suffix : str, default="_TESTFOLDS.csv"
        String added to dataset name to load.

    Returns
    -------
    list of dictionaries of dictionaries
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
    estimators : list of str
        List of estimators to search for.
    datasets : list of str, default = UCR.
        List of problem names to search for. Default is to look for the 112 UCR
        datasets listed in aeon.datasets.tsc_data_lists.
    default_only : boolean, default = True
        Whether to recover just the default test results, or 30 resamples. If false,
        values are averaged to get a 2D array.
    include_missing : boolean, default = False
        If a classifier does not have results for a given problem, either the whole
        problem is ignored when include_missing is False, or NaN.
    path : str, default https://timeseriesclassification.com/results/ReferenceResults/
        Path where to read results from, default to tsc.com.

    Returns
    -------
    2D numpy array
        Each column is a results for a classifier, each row a dataset.
    if include_missing == false, returns names: an aligned list of names of included.

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
            else:
                r[i] = False
        if include:
            all_res.append(r)
            names.append(d)

    if include_missing:
        return np.array(all_res)
    else:
        return np.array(all_res), names


def _get_published_results(
    directory, classifiers, resamples, suffix, default_only, header, n_data
):
    path = (
        "https://timeseriesclassification.com/results/PublishedResults/"
        + directory
        + "/"
    )
    estimators = classifiers
    all_results = {}
    for cls in estimators:
        url = path + cls + suffix
        try:
            data = pd.read_csv(url, header=header)
        except Exception:
            print(" Error trying to load from url", url)  # noqa
            print(" Check results for ", cls, " are on the website")  # noqa
            raise
        problems = data.iloc[:, 0].tolist()
        results = data.iloc[:, 1:].to_numpy()
        cls_results = np.zeros(shape=len(problems))
        if results.shape[1] != resamples:
            results = results[:, :resamples]
        for i in range(len(problems)):
            if default_only:
                cls_results[i] = results[i][0]
            else:
                cls_results[i] = np.nanmean(results[i])
        all_results[cls] = cls_results
    arrays = [v[:n_data] for v in all_results.values()]
    data_array = np.stack(arrays, axis=-1)
    return data_array


# Classifiers used in the original 2017 univariate TSC bake off
uni_classifiers_2017 = {
    "ACF": 0,
    "BOSS": 1,
    "CID_DTW": 2,
    "CID_ED": 3,
    "DDTW_R1_1NN": 4,
    "DDTW_Rn_1NN": 5,
    "DTW_F": 6,
    "EE": 7,
    "ERP_1NN": 8,
    "Euclidean_1NN": 9,
    "FlatCOTE": 10,
    "FS": 11,
    "LCSS_1NN": 12,
    "LPS": 13,
    "LS": 14,
    "MSM_1NN": 15,
    "PS": 16,
    "RotF": 17,
    "SAXVSM": 18,
    "ST": 19,
    "TSBF": 20,
    "TSF": 21,
    "TWE_1NN": 22,
    "WDDTW_1NN": 23,
    "WDTW_1NN": 24,
}

# Classifiers used in the 2021 multivariate TSC bake off
multi_classifiers_2021 = {
    "CBOSS": 0,
    "CIF": 1,
    "DTW_D": 2,
    "DTW_I": 3,
    "gRSF": 4,
    "HIVE-COTEv1": 5,
    "ResNet": 6,
    "RISE": 7,
    "ROCKET": 8,
    "STC": 9,
    "TSF": 10,
}

uni_classifiers_2023 = {
    "Arsenal": 0,
    "BOSS": 1,
    "CIF": 2,
    "CNN": 3,
    "Catch22": 4,
    "DrCIF": 5,
    "EE": 6,
    "FreshPRINCE": 7,
    "HC1": 8,
    "HC2": 9,
    "Hydra-MR": 10,
    "Hydra": 11,
    "InceptionT": 12,
    "Mini-R": 13,
    "MrSQM": 14,
    "Multi-R": 15,
    "PF": 16,
    "RDST": 17,
    "RISE": 18,
    "ROCKET": 19,
    "RSF": 20,
    "RSTSF": 21,
    "ResNet": 22,
    "STC": 23,
    "ShapeDTW": 24,
    "Signatures": 25,
    "TDE": 26,
    "TS-CHIEF": 27,
    "TSF": 28,
    "TSFresh": 29,
    "WEASEL-D": 30,
    "WEASEL": 31,
    "cBOSS": 32,
    "1NN-DTW": 33,
}


def get_bake_off_2017_results(default_only=True):
    """Fetch all the results of the 2017 univariate TSC bake off [1]_ from tsc.com.

    Basic utility function to recover legacy results. Loads results for 85
    univariate UCR data sets for all the classifiers listed in ``classifiers_2017``.
    Can load either the
    default train/test split, or the results averaged over 100 resamples.

    Parameters
    ----------
    default_only : boolean, default = True
        Whether to return the results for the default train/test split, or results
        averaged over resamples.

    Returns
    -------
    2D numpy array
        Each column is a results for a classifier, each row a dataset.

    References
    ----------
    .. [1] A Bagnall, J Lines, A Bostrom, J Large, E Keogh, "The great time series
    classification bake off: a review and experimental evaluation of recent
    algorithmic advances", Data mining and knowledge discovery 31, 606-660, 2017.

    Examples
    --------
    >>> from aeon.benchmarking import get_bake_off_2017_results, uni_classifiers_2017
    >>> from aeon.benchmarking import plot_critical_difference
    >>> default_results = get_bake_off_2017_results(default_only=True) # doctest: +SKIP
    >>> classifiers = ["MSM_1NN","LPS","TSBF","TSF","DTW_F","EE","BOSS","ST","FlatCOTE"]
    >>> # Get column positions of classifiers in results
    >>> cls = uni_classifiers_2017
    >>> index =[cls[key] for key in classifiers if key in cls]
    >>> selected =default_results[:,index] # doctest: +SKIP
    >>> plot = plot_critical_difference(selected, classifiers)# doctest: +SKIP
    >>> plot.show()# doctest: +SKIP
    >>> average_results = get_bake_off_2017_results(default_only=True) # doctest: +SKIP
    >>> selected =average_results[:,index] # doctest: +SKIP
    >>> plot = plot_critical_difference(selected, cls)# doctest: +SKIP
    >>> plot.show()# doctest: +SKIP
    """
    return _get_published_results(
        directory="Bakeoff2017",
        classifiers=uni_classifiers_2017,
        resamples=100,
        suffix=".csv",
        default_only=default_only,
        header=None,
        n_data=85,
    )


def get_bake_off_2021_results(default_only=True):
    """Pull down all the results of the 2020 multivariate bake off [1]_ from tsc.com.

    Basic utility function to recover legacy results. Loads results for 26 tsml
    data sets for all the classifiers listed in ``classifiers_2021``. Can load either
    the default train/test split, or the results averaged over 30 resamples.

    Parameters
    ----------
    default_only : boolean, default = True
        Whether to return the results for the default train/test split, or results
        averaged over resamples.

    Returns
    -------
    2D numpy array
        Each column is a results for a classifier, each row a dataset.

    References
    ----------
    .. [1] AP Ruiz, M Flynn, J Large, M Middlehurst, A Bagnall, "The great multivariate
    time series classification bake off: a review and experimental evaluation of
    recent algorithmic advances", Data mining and knowledge discovery 35, 401-449, 2021.

    Examples
    --------
    >>> from aeon.benchmarking import get_bake_off_2021_results, multi_classifiers_2021
    >>> from aeon.benchmarking import plot_critical_difference
    >>> default_results = get_bake_off_2021_results(default_only=True) # doctest: +SKIP
    >>> cls = list(multi_classifiers_2021.keys()) # doctest: +SKIP
    >>> selected =default_results # doctest: +SKIP
    >>> plot = plot_critical_difference(selected, cls)# doctest: +SKIP
    >>> plot.show()# doctest: +SKIP
    >>> average_results = get_bake_off_2021_results(default_only=False) # doctest: +SKIP
    >>> selected =average_results # doctest: +SKIP
    >>> plot = plot_critical_difference(selected, cls)# doctest: +SKIP
    >>> plot.show()# doctest: +SKIP
    """
    return _get_published_results(
        directory="Bakeoff2021",
        classifiers=multi_classifiers_2021,
        resamples=30,
        suffix="_TESTFOLDS.csv",
        default_only=default_only,
        header="infer",
        n_data=26,
    )


def get_bake_off_2023_results(default_only=True):
    """Pull down all the results of the 2023 univariate bake off [1]_ from tsc.com.

    Basic utility function to recover legacy results. Loads results for 112 UCR/tsml
    data sets for all the classifiers listed in ``classifiers_2023``. Can load
    either the default train/test split, or the results averaged over 30 resamples.
    Please note this paper is under review, and there are more extensive results on
    new datasets we will make more generally avaiable once published.

    Parameters
    ----------
    default_only : boolean, default = True
        Whether to return the results for the default train/test split, or results
        averaged over resamples.

    Returns
    -------
    2D numpy array
        Each column is a results for a classifier, each row a dataset.

    References
    ----------
    .. [1] M Middlehurst, P Schaefer, A Bagnall, "Bake off redux: a review and
    experimental evaluation of recent time series classification algorithms",
    arXiv preprint arXiv:2304.13029, 2023.

    Examples
    --------
    >>> from aeon.benchmarking import get_bake_off_2023_results, uni_classifiers_2023
    >>> from aeon.benchmarking import plot_critical_difference
    >>> default_results = get_bake_off_2023_results(default_only=True) # doctest: +SKIP
    >>> classifiers = ["HC2","MR-Hydra","InceptionT", "FreshPRINCE","WEASEL-D","RDST"]
    >>> # Get column positions of classifiers in results
    >>> cls = uni_classifiers_2023
    >>> index =[cls[key] for key in classifiers if key in cls]
    >>> selected =default_results[:,index] # doctest: +SKIP
    >>> plot = plot_critical_difference(selected, classifiers)# doctest: +SKIP
    >>> plot.show()# doctest: +SKIP
    >>> average_results = get_bake_off_2023_results(default_only=False) # doctest: +SKIP
    >>> selected =average_results[:,index] # doctest: +SKIP
    >>> plot = plot_critical_difference(selected, classifiers)# doctest: +SKIP
    >>> plot.show()# doctest: +SKIP


    """
    return _get_published_results(
        directory="Bakeoff2023",
        classifiers=uni_classifiers_2023,
        resamples=30,
        suffix="_TESTFOLDS.csv",
        default_only=default_only,
        header="infer",
        n_data=112,
    )
