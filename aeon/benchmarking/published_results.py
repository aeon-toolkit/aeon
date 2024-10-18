"""Functions to load published results."""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "get_bake_off_2017_results",
    "get_bake_off_2021_results",
    "get_bake_off_2023_results",
]

import numpy as np
import pandas as pd

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
    >>> from aeon.visualisation import plot_critical_difference
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
    >>> from aeon.visualisation import plot_critical_difference
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


# Classifiers used in the 2023 univariate TSC bake off
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
    >>> from aeon.visualisation import plot_critical_difference
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
