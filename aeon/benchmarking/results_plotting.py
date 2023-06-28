# -*- coding: utf-8 -*-
"""Plotting tools for estimator results."""
__all__ = [
    "plot_critical_difference",
    "plot_boxplot_median",
    "plot_scatter_single",
    "plot_scatter",
]

__author__ = ["dguijo"]

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


def plot_scatter_single(
    results,
    method,
    dataset,
    title=None,
    y_min=None,
    y_max=None,
):
    """Plot a scatter that compares actual and predicted values for a given dataset.

    Parameters
    ----------
    results: np.array
        Scores (either accuracies or errors) of dataset x strategy
    method: str
        Method's name.
    dataset: str
        Dataset's name.
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
    pass


def plot_scatter(
    results,
    method_A,
    method_B,
    title=None,
    y_min=None,
    y_max=None,
):
    """Plot a scatter that compares datasets' results achieved by two methods.

    Parameters
    ----------
    results: np.array
        Scores (either accuracies or errors) of dataset x strategy.
    method_A: str
        Method name of the first approach.
    method_B: str
        Method name of the second approach.
    dataset: str
        Dataset's name.
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
    fig = plt.figure(figsize=(10, 6))

    differences = [0 if i - j == 0 else (1 if i - j > 0 else -1) for i, j in results]

    min_value = results.min() * 0.97
    max_value = results.max() * 1.03

    x, y = [min_value, max_value], [min_value, max_value]
    plt.plot(x, y, color="black", alpha=0.5, zorder=1)

    plot = sns.scatterplot(
        x=results[:, 1],  # second method
        y=results[:, 0],  # first method
        hue=differences,
        palette="pastel",
        zorder=2,
    )

    # Compute the W, T, and L per methods
    wins_A = sum(i == 1 for i in differences)
    ties_A = sum(i == 0 for i in differences)
    losses_A = sum(i == -1 for i in differences)

    wins_B = sum(i == -1 for i in differences)
    ties_B = sum(i == 0 for i in differences)
    losses_B = sum(i == 1 for i in differences)

    # Setting x and y limits
    plot.set_ylim(min_value, max_value)
    plot.set_xlim(min_value, max_value)

    # Remove legend
    plot.get_legend().remove()

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    min_value_text = results.min()
    max_value_text = results.max()

    # Setting labels for x and y axis
    plot.set_xlabel(f"{method_B} accuracy")
    plot.set_ylabel(f"{method_A} accuracy")

    # Setting text with W, T and L for each method
    plt.text(
        min_value_text,
        max_value_text * 0.98,
        f"{method_A} wins here\n[{wins_A}W, {ties_A}T, {losses_A}L]",
        fontsize=13,
        va="top",
        ha="left",
        ma="center",
        bbox=props,
        clip_on=True,
        color="darkseagreen",
        fontweight="bold",
    )

    plt.text(
        max_value_text,
        min_value_text * 1.02,
        f"{method_B} wins here\n[{wins_B}W, {ties_B}T, {losses_B}L]",
        fontsize=13,
        va="bottom",
        ha="right",
        ma="center",
        bbox=props,
        clip_on=True,
        color="cornflowerblue",
        fontweight="bold",
    )

    return fig
