# -*- coding: utf-8 -*-
"""Plotting tools for estimator results."""
__all__ = ["plot_critical_difference", "plot_boxplot_median"]

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from aeon.benchmarking._critical_difference import plot_critical_difference


def plot_boxplot_median(
    results,
    labels,
    plot_type="violin",
    outliers=True,
    title=None,
    y_min=None,
    y_max=None,
):
    """
    Plot a box plot of distributions from the median.

    Each row of results is an independent experiment for each element in names. This
    function works out the deviation from the median for each row, then plots a
    boxplot variant of each column.

    Parameters
    ----------
    results: np.array
        Scores (either accuracies or errors) of dataset x strategy
    labels: list of estimators
        List with names of the estimators
    plot_type: str, default = "violin"
        This function can create two sort of distribution plots: "violin", "swarm",
        "boxplot". "violin" plot features a kernel density estimation of the underlying
        distribution. "swarm" draws a categorical scatterplot with points adjusted to be
        non-overlapping.
    outliers: bool, default = True
        Only applies when plot_type is "boxplot".
    title: str, default = None
        Title to be shown in the top of the plot.
    y_min: float, default = None
        Min value for the y_axis of the plot.
    y_max: float, default = None
        Max value for the y_axis of the plot.

    Returns
    -------
    fig: matplotlib.figure
        Figure created.
    """
    # Obtains deviation from median for each independent experiment.
    medians = np.median(results, axis=1)
    deviation_from_median = results / (results + medians[:, np.newaxis])

    fig = plt.figure(figsize=(10, 6))

    # Plots violin or boxplots
    if plot_type == "violin":
        plot = sns.violinplot(
            data=deviation_from_median,
            lw=0.2,
            palette="pastel",
            bw=0.3,
        )
    elif plot_type == "boxplot":
        plot = sns.boxplot(
            data=deviation_from_median,
            palette="pastel",
            showfliers=outliers,
        )
    elif plot_type == "swarm":
        plot = sns.swarmplot(
            data=deviation_from_median,
            lw=0.2,
            palette="pastel",
        )
    elif plot_type == "strip":
        plot = sns.stripplot(
            data=deviation_from_median,
            lw=0.2,
            palette="pastel",
        )

    # Modifying limits for y-axis.
    if y_min is None and (
        (plot_type == "boxplot" and outliers) or (plot_type != "boxplot")
    ):
        y_min = np.around(np.amin(deviation_from_median) - 0.05, 2)

    if y_max is None and (
        (plot_type == "boxplot" and outliers) or (plot_type != "boxplot")
    ):
        y_max = np.around(np.amax(deviation_from_median) + 0.05, 2)

    plot.set_ylim(y_min, y_max)

    # Setting labels for x-axis.
    plot.set_xticklabels(labels, rotation=45, ha="right")

    # Setting title if provided.
    if title is not None:
        plot.set_title(rf"{title}")

    return fig
