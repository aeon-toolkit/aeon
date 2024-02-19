"""Functions for plotting results boxplot diagrams."""

__maintainer__ = []

__all__ = [
    "plot_boxplot_median",
]

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies


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
        This function can create four sort of distribution plots: "violin", "swarm",
        "boxplot" or "strip". "violin" plot features a kernel density estimation of the
        underlying distribution. "swarm" draws a categorical scatterplot with points
        adjusted to be non-overlapping. "strip" draws a categorical scatterplot using
        jitter to reduce overplotting.
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
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Example
    -------
    >>> from aeon.visualisation import plot_boxplot_median
    >>> from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    >>> methods = ["IT", "WEASEL-Dilation", "HIVECOTE2", "FreshPRINCE"]
    >>> results = get_estimator_results_as_array(estimators=methods) # doctest: +SKIP
    >>> plot = plot_boxplot_median(results[0], methods) # doctest: +SKIP
    >>> plot.show() # doctest: +SKIP
    >>> plot.savefig("boxplot.pdf") # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Obtains deviation from median for each independent experiment.
    medians = np.median(results, axis=1)
    sum_results_medians = results + medians[:, np.newaxis]

    deviation_from_median = np.divide(
        results,
        sum_results_medians,
        out=np.zeros_like(results),
        where=sum_results_medians != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 6), layout="tight")

    # Plots violin or boxplots
    if plot_type == "violin":
        plot = sns.violinplot(
            data=deviation_from_median,
            linewidth=0.2,
            palette="pastel",
            bw_method=0.3,
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
            linewidth=0.2,
            palette="pastel",
        )
    elif plot_type == "strip":
        plot = sns.stripplot(
            data=deviation_from_median,
            linewidth=0.2,
            palette="pastel",
        )
    else:
        raise ValueError(
            "plot_type must be one of 'violin', 'boxplot', 'swarm' or 'strip'."
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

    ax.set_ylim(y_min, y_max)

    # Setting labels for x-axis. Rotate only if labels are too long.
    ax.set_xticks(np.arange(len(labels)))
    label_lengths = np.array([len(i) for i in labels])
    if (sum(label_lengths) > 40) or (max(label_lengths[:-1] + label_lengths[1:]) > 20):
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(labels)

    # Setting title if provided.
    if title is not None:
        plot.set_title(rf"{title}")

    return fig, ax
