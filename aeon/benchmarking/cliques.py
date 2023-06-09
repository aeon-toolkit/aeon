# -*- coding: utf-8 -*-
"""Functions to find cliques in results."""
import numpy as np
from statsmodels.tests import wilcoxon


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
    pass
