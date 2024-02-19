"""Functions to compute stats and get p-values."""

__maintainer__ = []

__all__ = ["check_friedman", "nemenyi_test", "wilcoxon_test"]

import warnings

import numpy as np
from scipy.stats import distributions, find_repeats, wilcoxon

from aeon.benchmarking.utils import get_qalpha


def check_friedman(ranks):
    """
    Check whether Friedman test is significant.

    Parameters
    ----------
    ranks : np.array
        Rank of estimators on datasets, shape (n_estimators, n_datasets).

    Returns
    -------
    float
      p-value of the test.
    """
    n_datasets, n_estimators = ranks.shape

    if n_estimators < 3:
        raise ValueError(
            "At least 3 sets of measurements must be given for Friedmann test, "
            f"got {n_estimators}."
        )

    # calculate c to correct chisq for ties:
    ties = 0
    for i in range(n_datasets):
        replist, repnum = find_repeats(ranks[i])
        for t in repnum:
            ties += t * (t * t - 1)
    c = 1 - ties / (n_estimators * (n_estimators * n_estimators - 1) * n_datasets)

    ssbn = np.sum(ranks.sum(axis=0) ** 2)
    chisq = (
        12.0 / (n_estimators * n_datasets * (n_estimators + 1)) * ssbn
        - 3 * n_datasets * (n_estimators + 1)
    ) / c
    p_value = distributions.chi2.sf(chisq, n_estimators - 1)
    return p_value


def nemenyi_test(ordered_avg_ranks, n_datasets, alpha):
    """
    Find cliques using post hoc Nemenyi test.

    Parameters
    ----------
    ordered_avg_ranks : np.array
        Average ranks of estimators.
    n_datasets : int
        Mumber of datasets.
    alpha : float
        alpha level for Nemenyi test.

    Returns
    -------
    list of lists
        List of cliques. A clique is a group of estimators within which there is no
        significant difference.
    """
    n_estimators = len(ordered_avg_ranks)
    qalpha = get_qalpha(alpha)
    # calculate critical difference with Nemenyi
    cd = qalpha[n_estimators] * np.sqrt(
        n_estimators * (n_estimators + 1) / (6 * n_datasets)
    )
    # compute statistically similar cliques
    cliques = np.tile(ordered_avg_ranks, (n_estimators, 1)) - np.tile(
        np.vstack(ordered_avg_ranks.T), (1, n_estimators)
    )
    cliques[cliques < 0] = np.inf
    cliques = cliques < cd

    return cliques


def wilcoxon_test(results, labels, lower_better=False):
    """
    Perform Wilcoxon test.

    Parameters
    ----------
    results : np.array
        Results of estimators on datasets

    lower_better : bool, default = False
        Indicates whether smaller is better for the results in scores. For example,
        if errors are passed instead of accuracies, set ``lower_better`` to ``True``.

    Returns
    -------
    np.array
        p-values of Wilcoxon sign rank test.
    """
    n_estimators = results.shape[1]

    p_values = np.eye(n_estimators)

    for i in range(n_estimators - 1):
        for j in range(i + 1, n_estimators):
            # if the difference is zero, the p-value is 1
            if np.all(results[:, i] == results[:, j]):
                p_values[i, j] = 1
                # raise warning
                warnings.warn(
                    f"Estimators {labels[i]} and {labels[j]} have the same performance"
                    "on all datasets. This may cause problems when forming cliques.",
                    stacklevel=2,
                )
            else:
                p_values[i, j] = wilcoxon(
                    results[:, i],
                    results[:, j],
                    zero_method="wilcox",
                    alternative="less" if lower_better else "greater",
                )[1]
    return p_values
