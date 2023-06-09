# -*- coding: utf-8 -*-
"""Functions to find cliques in results."""
import numpy as np
from scipy.stats import distributions, find_repeats, wilcoxon


def friedman_test(n_estimators: int, n_datasets: int, ranked_data: np.ndarray) -> float:
    """
    Perform the Friedman test for any difference in rank for multiple estimators.

    Larger parts of code copied from scipy.

    Parameters
    ----------
    n_estimators : int
      number of strategies to evaluate
    n_datasets : int
      number of datasets classified per strategy
    ranked_data : np.array shape (n_datasets, n_estimators)
      rank of strategy on dataset

    Returns
    -------
        p : float
        p value for Friedman test
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
    p = distributions.chi2.sf(chisq, n_estimators - 1)
    return p


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
    >>> from aeon.benchmarking.cliques import pairwise_wilcoxon
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
