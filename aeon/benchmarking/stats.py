"""Functions to compute stats and get p-values."""

__maintainer__ = []
__all__ = ["check_friedman", "nemenyi_test", "wilcoxon_test"]

import warnings

import numpy as np
from scipy.stats import distributions, find_repeats, wilcoxon


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
    qalpha = _get_qalpha(alpha)
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


def _get_qalpha(alpha: float):
    """Get the alpha value for post hoc Nemenyi."""
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
        raise Exception("alpha must be 0.01, 0.05 or 0.1")
    return qalpha


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
                ).pvalue
    return p_values
