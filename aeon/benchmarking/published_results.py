"""Functions to load published results."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "load_classification_bake_off_2017_results",
    "load_classification_bake_off_2021_results",
    "load_classification_bake_off_2023_results",
]

from aeon.benchmarking.results_loaders import _load_to_dict, _results_dict_to_array
from aeon.datasets.tsc_datasets import (
    multivariate_equal_length,
    univariate2015,
    univariate_equal_length,
)


def load_classification_bake_off_2017_results(
    num_resamples=100, as_array=False, ignore_nan=False
):
    """Fetch all the results of the 2017 univariate TSC bake off.

    Basic utility function to recover legacy results from [1]_. Loads results for 85
    univariate UCR data sets for  classifiers used in the publication. Can load either
    the default train/test split, or the resampled results up to 100 resamples.

    Parameters
    ----------
    num_resamples : int or None, default=1
        The number of data resamples to return scores for. The first resample
        is the default train/test split for the dataset.
        For 1, only the score for the default train/test split of the dataset is
        returned.
        For 2 or more, a np.ndarray of scores for all resamples up to num_resamples are
        returned.
        If None, the scores of all resamples are returned.

        If as_array is true, the scores are averaged instead of being returned as a
        np.ndarray.
    as_array : bool, default=False
        If True, return the results as a tuple containing a np.ndarray of (averaged)
        scores for each classifier. Also returns a list of dataset names for each
        row of the np.ndarray, and classifier names for each column.
    ignore_nan : bool, default=False
        Ignore the error raised when NaN values are present in the results. Ignores
        NaN values when averaging when as_array is True.

    Returns
    -------
    results: dict or tuple
        Dictionary with estimator name keys containing another dictionary.
        Sub-dictionary consists of dataset name keys and contains of scores for each
        dataset.
        If as_array is true, instead returns a tuple of: An array of scores. Each
        column is a results for a classifier, each row a dataset. A list of dataset
        names for each row. A list of classifier names for each column.

    References
    ----------
    .. [1] A Bagnall, J Lines, A Bostrom, J Large, E Keogh, "The great time series
        classification bake off: a review and experimental evaluation of recent
        algorithmic advances", Data mining and knowledge discovery 31, 606-660, 2017.

    Examples
    --------
    >>> from aeon.benchmarking.published_results import (
    ...     load_classification_bake_off_2017_results
    ... )
    >>> from aeon.visualisation import plot_critical_difference
    >>> # Load the results
    >>> results, data, cls = load_classification_bake_off_2023_results(
    ...     num_resamples=100, as_array=True
    ... )  # doctest: +SKIP
    >>> # Select a subset of classifiers
    >>> cls = ["MSM_1NN","TSF","DTW_F","EE","BOSS","ST","FlatCOTE"] # doctest: +SKIP
    >>> index = [cls.index(i) for i in cls] # doctest: +SKIP
    >>> selected = results[:,index]  # doctest: +SKIP
    >>> # Plot the critical difference diagram
    >>> plot = plot_critical_difference(selected, cls)  # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    """
    path = "https://timeseriesclassification.com/results/PublishedResults/Bakeoff2017/"
    classifiers = [
        "ACF",
        "BOSS",
        "CID_DTW",
        "CID_ED",
        "DDTW_R1_1NN",
        "DDTW_Rn_1NN",
        "DTW_F",
        "EE",
        "ERP_1NN",
        "Euclidean_1NN",
        "FlatCOTE",
        "FS",
        "LCSS_1NN",
        "LPS",
        "LS",
        "MSM_1NN",
        "PS",
        "RotF",
        "SAXVSM",
        "ST",
        "TSBF",
        "TSF",
        "TWE_1NN",
        "WDDTW_1NN",
        "WDTW_1NN",
    ]
    res = _load_to_dict(
        path=path,
        estimators=classifiers,
        datasets=univariate2015,
        num_resamples=num_resamples,
        file_suffix=".csv",
        est_alias=False,
        csv_header=None,
        ignore_nan=True,
    )
    if as_array:
        res, datasets = _results_dict_to_array(res, classifiers, univariate2015, False)
        return res, datasets, classifiers
    return res


def load_classification_bake_off_2021_results(num_resamples=30, as_array=False):
    """Pull down all the results of the 2021 multivariate bake off.

    Basic utility function to recover legacy results from [1]_. Loads results for 26
    tsml data sets for classifiers used in the publication. Can load either
    the default train/test split, or the resampled results up to 30 resamples.

    Parameters
    ----------
    num_resamples : int or None, default=1
        The number of data resamples to return scores for. The first resample
        is the default train/test split for the dataset.
        For 1, only the score for the default train/test split of the dataset is
        returned.
        For 2 or more, a np.ndarray of scores for all resamples up to num_resamples are
        returned.
        If None, the scores of all resamples are returned.

        If as_array is true, the scores are averaged instead of being returned as a
        np.ndarray.
    as_array : bool, default=False
        If True, return the results as a tuple containing a np.ndarray of (averaged)
        scores for each classifier. Also returns a list of dataset names for each
        row of the np.ndarray, and classifier names for each column.

    Returns
    -------
    results: dict or tuple
        Dictionary with estimator name keys containing another dictionary.
        Sub-dictionary consists of dataset name keys and contains of scores for each
        dataset.
        If as_array is true, instead returns a tuple of: An array of scores. Each
        column is a results for a classifier, each row a dataset. A list of dataset
        names for each row. A list of classifier names for each column.

    References
    ----------
    .. [1] AP Ruiz, M Flynn, J Large, M Middlehurst, A Bagnall, "The great multivariate
        time series classification bake off: a review and experimental evaluation of
        recent algorithmic advances", Data mining and knowledge discovery 35, 401-449,
        2021.

    Examples
    --------
    >>> from aeon.benchmarking.published_results import (
    ...     load_classification_bake_off_2021_results
    ... )
    >>> from aeon.visualisation import plot_critical_difference
    >>> # Load the results
    >>> results, data, cls = load_classification_bake_off_2023_results(
    ...     num_resamples=30, as_array=True
    ... )  # doctest: +SKIP
    >>> # Plot the critical difference diagram
    >>> plot = plot_critical_difference(results, cls)  # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    """
    path = "https://timeseriesclassification.com/results/PublishedResults/Bakeoff2021/"
    classifiers = [
        "CBOSS",
        "CIF",
        "DTW_D",
        "DTW_I",
        "gRSF",
        "HIVE-COTEv1",
        "ResNet",
        "RISE",
        "ROCKET",
        "STC",
        "TSF",
    ]
    res = _load_to_dict(
        path=path,
        estimators=classifiers,
        datasets=multivariate_equal_length,
        num_resamples=num_resamples,
        file_suffix="_TESTFOLDS.csv",
        est_alias=False,
    )
    if as_array:
        res, datasets = _results_dict_to_array(
            res, classifiers, multivariate_equal_length, False
        )
        return res, datasets, classifiers
    return res


def load_classification_bake_off_2023_results(num_resamples=30, as_array=False):
    """Pull down all the results of the 2023 univariate bake off.

    Basic utility function to recover legacy results from [1]_. Loads results for 112
    UCR/tsml data sets for classifiers used in the publication. Can load either
    the default train/test split, or the resampled results up to 30 resamples.

    Parameters
    ----------
    num_resamples : int or None, default=1
        The number of data resamples to return scores for. The first resample
        is the default train/test split for the dataset.
        For 1, only the score for the default train/test split of the dataset is
        returned.
        For 2 or more, a np.ndarray of scores for all resamples up to num_resamples are
        returned.
        If None, the scores of all resamples are returned.

        If as_array is true, the scores are averaged instead of being returned as a
        np.ndarray.
    as_array : bool, default=False
        If True, return the results as a tuple containing a np.ndarray of (averaged)
        scores for each classifier. Also returns a list of dataset names for each
        row of the np.ndarray, and classifier names for each column.

    Returns
    -------
    results: dict or tuple
        Dictionary with estimator name keys containing another dictionary.
        Sub-dictionary consists of dataset name keys and contains of scores for each
        dataset.
        If as_array is true, instead returns a tuple of: An array of scores. Each
        column is a results for a classifier, each row a dataset. A list of dataset
        names for each row. A list of classifier names for each column.

    References
    ----------
    .. [1] M Middlehurst, P Schaefer, A Bagnall, "Bake off redux: a review and
        experimental evaluation of recent time series classification algorithms",
        arXiv preprint arXiv:2304.13029, 2023.

    Examples
    --------
    >>> from aeon.benchmarking.published_results import (
    ...     load_classification_bake_off_2023_results
    ... )
    >>> from aeon.visualisation import plot_critical_difference
    >>> # Load the results
    >>> results, data, cls = load_classification_bake_off_2023_results(
    ...     num_resamples=30, as_array=True
    ... )  # doctest: +SKIP
    >>> # Select a subset of classifiers
    >>> cls = ["HC2","MR-Hydra","InceptionT","FreshPRINCE","RDST"] # doctest: +SKIP
    >>> index = [cls.index(i) for i in cls] # doctest: +SKIP
    >>> selected = results[:,index]  # doctest: +SKIP
    >>> # Plot the critical difference diagram
    >>> plot = plot_critical_difference(selected, cls)  # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    """
    path = "https://timeseriesclassification.com/results/PublishedResults/Bakeoff2023/"
    classifiers = [
        "Arsenal",
        "BOSS",
        "CIF",
        "CNN",
        "Catch22",
        "DrCIF",
        "EE",
        "FreshPRINCE",
        "HC1",
        "HC2",
        "Hydra-MR",
        "Hydra",
        "InceptionT",
        "Mini-R",
        "MrSQM",
        "Multi-R",
        "PF",
        "RDST",
        "RISE",
        "ROCKET",
        "RSF",
        "RSTSF",
        "ResNet",
        "STC",
        "ShapeDTW",
        "Signatures",
        "TDE",
        "TS-CHIEF",
        "TSF",
        "TSFresh",
        "WEASEL-D",
        "WEASEL",
        "cBOSS",
        "1NN-DTW",
    ]
    res = _load_to_dict(
        path=path,
        estimators=classifiers,
        datasets=univariate_equal_length,
        num_resamples=num_resamples,
        file_suffix="_TESTFOLDS.csv",
        est_alias=False,
    )
    if as_array:
        res, datasets = _results_dict_to_array(
            res, classifiers, univariate_equal_length, False
        )
        return res, datasets, classifiers
    return res
