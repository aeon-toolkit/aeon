# -*- coding: utf-8 -*-
"""Statistical tests for the benchmarking module."""

import numpy as np
from scipy.stats import distributions, find_repeats, wilcoxon

__all__ = ["pairwise_wilcoxon", "friedman_test", "calculate_nemenyi_q_critical"]


def friedman_test(n_estimators, n_datasets, ranked_data):
    """
    Check whether Friedman test is significant.

    Larger parts of code copied from scipy.

    Arguments
    ---------
    n_estimators : int
      number of strategies to evaluate
    n_datasets : int
      number of datasets classified per strategy
    ranked_data : np.array (shape: n_estimators * n_datasets)
      rank of strategy on dataset

    Returns
    -------
    p_val : float
      Result of Friedman test.
    """
    if n_estimators < 3:
        raise ValueError(
            "At least 3 sets of measurements must be given for Friedmann test, "
            f"got {n_estimators}."
        )

    # calculate c to correct chisq for ties:
    ties = 0
    for i in range(n_datasets):
        replist, repnum = find_repeats(ranked_data[i])
        for t in repnum:
            ties += t * (t * t - 1)
    c = 1 - ties / (n_estimators * (n_estimators * n_estimators - 1) * n_datasets)

    ssbn = np.sum(ranked_data.sum(axis=0) ** 2)
    chisq = (
        12.0 / (n_estimators * n_datasets * (n_estimators + 1)) * ssbn
        - 3 * n_datasets * (n_estimators + 1)
    ) / c
    p_val = distributions.chi2.sf(chisq, n_estimators - 1)
    return p_val


def pairwise_wilcoxon(results: np.ndarray) -> np.ndarray:
    """Perform all pairwise tests on a set of results.

    Parameters
    ----------
        results: np.ndarray shape (n_problems, n_estimators).
            Results to perform tests on. Each row stores results for a single
            problem. Each column stores results for a single estimator.

    Returns
    -------
        p_values: (n_estimators, n_estimators) array of unadjusted p values,
        with zeros on the diagonal.

    Example
    -------
    >>> from aeon.benchmarking.results_tests import pairwise_wilcoxon
    >>> results = np.array([[0.8, 0.75, 0.4], [0.7, 0.6, 0.35], [0.65, 0.9, 0.35]])
    >>> pairwise_wilcoxon(results)
    array([[0.  , 1.  , 0.25],
           [1.  , 0.  , 0.25],
           [0.25, 0.25, 0.  ]])

    """
    n, m = results.shape
    p_values = np.zeros((m, m))

    for i in range(m):
        for j in range(i, m):
            if i != j:
                _, p_value = wilcoxon(results[:, i], results[:, j])
                p_values[i, j] = p_value
                p_values[j, i] = p_value

    return p_values


def find_cliques(results: np.ndarray):
    """Find cliques within which there is no critical difference."""


def calculate_nemenyi_q_critical(
    n_estimators: int, n_datasets: int, alpha: float = 0.05
):
    """
    Calculate the critical value for the q statistic in the Nemenyi post-hoc test.

    Parameters
    ----------
        n_estimators : int. Number of treatments or groups.
        n_datasets : int/ Total number of observations or sample size.
        alpha : float. Significance level (default: 0.05).

    Returns
    -------
        float: The critical value for the q statistic.
    """
    if alpha == 0.01:
        qalpha = [
            0.000,
            2.576,
            2.913,
            3.113,
            3.255,
            3.364,
            3.452,
            3.526,
            3.590,
            3.646,
            3.696,
            3.741,
            3.781,
            3.818,
            3.853,
            3.884,
            3.914,
            3.941,
            3.967,
            3.992,
            4.015,
            4.037,
            4.057,
            4.077,
            4.096,
            4.114,
            4.132,
            4.148,
            4.164,
            4.179,
            4.194,
            4.208,
            4.222,
            4.236,
            4.249,
            4.261,
            4.273,
            4.285,
            4.296,
            4.307,
            4.318,
            4.329,
            4.339,
            4.349,
            4.359,
            4.368,
            4.378,
            4.387,
            4.395,
            4.404,
            4.412,
            4.420,
            4.428,
            4.435,
            4.442,
            4.449,
            4.456,
        ]

    elif alpha == 0.05:
        qalpha = [
            0.000,
            1.960,
            2.344,
            2.569,
            2.728,
            2.850,
            2.948,
            3.031,
            3.102,
            3.164,
            3.219,
            3.268,
            3.313,
            3.354,
            3.391,
            3.426,
            3.458,
            3.489,
            3.517,
            3.544,
            3.569,
            3.593,
            3.616,
            3.637,
            3.658,
            3.678,
            3.696,
            3.714,
            3.732,
            3.749,
            3.765,
            3.780,
            3.795,
            3.810,
            3.824,
            3.837,
            3.850,
            3.863,
            3.876,
            3.888,
            3.899,
            3.911,
            3.922,
            3.933,
            3.943,
            3.954,
            3.964,
            3.973,
            3.983,
            3.992,
            4.001,
            4.009,
            4.017,
            4.025,
            4.032,
            4.040,
            4.046,
        ]
    elif alpha == 0.1:
        qalpha = [
            0.000,
            1.645,
            2.052,
            2.291,
            2.460,
            2.589,
            2.693,
            2.780,
            2.855,
            2.920,
            2.978,
            3.030,
            3.077,
            3.120,
            3.159,
            3.196,
            3.230,
            3.261,
            3.291,
            3.319,
            3.346,
            3.371,
            3.394,
            3.417,
            3.439,
            3.459,
            3.479,
            3.498,
            3.516,
            3.533,
            3.550,
            3.567,
            3.582,
            3.597,
            3.612,
            3.626,
            3.640,
            3.653,
            3.666,
            3.679,
            3.691,
            3.703,
            3.714,
            3.726,
            3.737,
            3.747,
            3.758,
            3.768,
            3.778,
            3.788,
            3.797,
            3.806,
            3.814,
            3.823,
            3.831,
            3.838,
            3.846,
        ]
        #
    else:
        raise ValueError(" Alpha must be 0.01, 0.05 or 0.1")
    return qalpha[n_estimators]

    # import numpy as np
    # from scipy.stats import chi2
    # Calculate the degrees of freedom for the chi-squared distribution
    # df = k * (k + 1) / 2
    # Calculate the critical value for the q statistic
    # q_critical = np.sqrt(2) * chi2.ppf(1 - alpha, df)
    # Adjust the critical value for the sample size
    # q_critical *= np.sqrt(k * (k + 1) / (6 * n))
    # return q_critical
