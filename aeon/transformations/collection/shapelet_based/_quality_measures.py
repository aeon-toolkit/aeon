"""Quality measures for shapelet evaluation.

This module contains numba-optimized quality measures for evaluating shapelet
quality in the RandomShapeletTransform.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def f_statistic(class0_distances, class1_distances):
    """Calculate the F-statistic for shapelet quality.

    The F-statistic measures the ratio of between-class variance to within-class
    variance. Higher values indicate better class separation.

    Parameters
    ----------
    class0_distances : np.ndarray
        Array of distances for the first class.
    class1_distances : np.ndarray
        Array of distances for the second class.

    Returns
    -------
    float
        The computed F-statistic. Returns np.inf if either class is empty or
        if there are insufficient degrees of freedom.

    Notes
    -----
    The F-statistic is calculated as:
        F = (SSB / df_between) / (SSW / df_within)
    where SSB is the between-class sum of squares and SSW is the within-class
    sum of squares.
    """
    if len(class0_distances) == 0 or len(class1_distances) == 0:
        return np.inf

    # Calculate means
    mean_class0 = np.mean(class0_distances)
    mean_class1 = np.mean(class1_distances)
    all_distances = np.concatenate((class0_distances, class1_distances))
    overall_mean = np.mean(all_distances)

    n0 = len(class0_distances)
    n1 = len(class1_distances)
    total_n = n0 + n1

    # Between-class sum of squares
    ssb = (
        n0 * (mean_class0 - overall_mean) ** 2 + n1 * (mean_class1 - overall_mean) ** 2
    )

    # Within-class sum of squares
    ssw = np.sum((class0_distances - mean_class0) ** 2) + np.sum(
        (class1_distances - mean_class1) ** 2
    )

    # Degrees of freedom
    df_between = 1
    df_within = total_n - 2

    # Avoid division by zero
    if df_within <= 0:
        return np.inf

    f_stat = (ssb / df_between) / (ssw / df_within)
    return f_stat
