# -*- coding: utf-8 -*-
"""Plotting tools for estimator results."""
__all__ = ["plot_critical_difference"]

import numpy as np

from aeon.benchmarking._critical_difference import plot_critical_difference
from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_boxplot_median(results: np.ndarray, names):
    """Plot a box plot of distributions from the median.

    Each row of results is an independent experiment for each element in names. This
    function works out the deviation from the median for each row, then plots a
    boxplot variant of each column.

    Parameters
    ----------
        results:
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(font_scale=1.2)
    sns.set_style("white")
    row_medians = np.median(results, axis=1)

    # Calculate the deviation from the median for each row
    deviations = results - row_medians[:, np.newaxis]
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    # g=sns.boxplot(data=df, ax=ax)
    # g=sns.stripplot(data=df, ax=ax)
    sns.violinplot(data=deviations, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30, ha="left")  # fontsize=16
    ax.set_xlabel("")
    ax.set_ylabel("Deviation from\n median accuracy")
    # ax.set_title("Median accuracy on new datasets", fontsize=18)
    sns.despine()
    plt.tight_layout()
    plt.savefig("median_accuracy_violin.pdf", bbox_inches="tight")
