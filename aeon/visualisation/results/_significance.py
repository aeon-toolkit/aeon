"""Function to compute and plot significance."""

__maintainer__ = []

__all__ = [
    "plot_significance",
]

import warnings

import numpy as np
from scipy.stats import rankdata

from aeon.benchmarking.stats import check_friedman, nemenyi_test, wilcoxon_test
from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_significance(
    scores,
    labels,
    alpha=0.1,
    lower_better=False,
    test="wilcoxon",
    correction="holm",
    fontsize=12,
    reverse=True,
    return_p_values=False,
):
    """
    Plot similar to CDDs, but allows the case where cliques can be deceiving.

    It is a visual representation of the results of a statistical test to compare the
    performance of different classifiers. The plot is based on the work of Demsar [1]_.
    The plot shows the average rank of each classifier and the average score of each
    method. The plot also shows the cliques of classifiers that are not significantly
    different from each other. The main difference against CDDs is that this plot
    allows the case where cliques can be deceiving, i.e. there is a gap where a method
    is significantly different from the rest of the clique. In the CDDs, the clique
    stop when there is a significant difference between the methods. In this plot,
    the clique continues until the end of the list of methods, showing a gap for the
    method with significant differences.

    Parameters
    ----------
    scores : np.array
        Array of shape (n_datasets, n_estimators) with the performance of each estimator
        in each dataset.
    labels : list of str
        List of length n_estimators with the name of each estimator.
    alpha : float, default=0.1
        The significance level used in the statistical test.
    lower_better : bool, default=False
        Whether lower scores are better than higher scores.
    test : str, default="wilcoxon"
        The statistical test to use. Available tests are "nemenyi" and "wilcoxon".
    correction : str, default="holm"
        The correction to apply to the p-values. Available corrections are "bonferroni",
        "holm" and None.
    fontsize : int, default=12
        The fontsize of the text in the plot.
    reverse : bool, default=True
        Whether to reverse the order of the labels.
    return_p_values : bool, default=False
        Whether to return the pairwise matrix of p-values.

    Returns
    -------
    fig : matplotlib.figure
        Figure created.
    ax : matplotlib.axes
        Axes of the figure.
    p_values : np.ndarray (optional)
        if return_p_values is True, returns a (n_estimators, n_estimators) matrix of
        unadjusted p values for the pairwise Wilcoxon sign rank test.

    References
    ----------
    .. [1] Demsar J., "Statistical comparisons of classifiers over multiple data sets."
    Journal of Machine Learning Research 7:1-30, 2006.

    Examples
    --------
    >>> from aeon.visualisation import plot_significance
    >>> from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    >>> methods = ["IT", "WEASEL-Dilation", "HIVECOTE2", "FreshPRINCE"]
    >>> results = get_estimator_results_as_array(estimators=methods) # doctest: +SKIP
    >>> plot = plot_significance(results[0], methods, alpha=0.1)\
        # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    >>> plot.savefig("significance_plot.pdf", bbox_inches="tight")  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    n_datasets, n_estimators = scores.shape
    if isinstance(test, str):
        test = test.lower()
    if isinstance(correction, str):
        correction = correction.lower()
    # Step 1: rank data: in case of ties average ranks are assigned
    if lower_better:  # low is good -> rank 1
        ranks = rankdata(scores, axis=1)
    else:  # assign opposite ranks
        ranks = rankdata(-1 * scores, axis=1)

    if return_p_values and test != "wilcoxon":
        raise ValueError(
            f"Cannot return p values for {test}, since it does "
            "not calculate p-values."
        )

    # Step 2: calculate average rank per estimator
    avg_ranks = ranks.mean(axis=0)
    # Sort labels and ranks
    ordered_labels_ranks = np.array(
        [(l, float(r)) for r, l in sorted(zip(avg_ranks, labels))], dtype=object
    )
    ordered_labels = np.array([la for la, _ in ordered_labels_ranks], dtype=str)
    ordered_avg_ranks = np.array([r for _, r in ordered_labels_ranks], dtype=np.float32)

    indices = [np.where(np.array(labels) == r)[0] for r in ordered_labels]

    ordered_scores = scores[:, indices]
    ordered_avg_scores = ordered_scores.mean(axis=0).flatten()

    # Step 3 : check whether Friedman test is significant
    p_value_friedman = check_friedman(ranks)
    # Step 4: If Friedman test is significant find cliques
    if p_value_friedman < alpha:
        if test == "nemenyi":
            cliques = nemenyi_test(ordered_avg_ranks, n_datasets, alpha)
        elif test == "wilcoxon":
            if correction == "bonferroni":
                adjusted_alpha = alpha / (n_estimators * (n_estimators - 1) / 2)
            elif correction == "holm":
                adjusted_alpha = alpha / (n_estimators - 1)
            elif correction is None:
                adjusted_alpha = alpha
            else:
                raise ValueError("correction available are None, Bonferroni and Holm.")
            p_values = wilcoxon_test(ordered_scores, ordered_labels, lower_better)
            cliques = _build_cliques(p_values > adjusted_alpha)
        else:
            raise ValueError("tests available are only nemenyi and wilcoxon.")
    # If Friedman test is not significant everything has to be one clique
    else:
        p_values = np.triu(np.ones((n_estimators, n_estimators)))
        cliques = [[1] * n_estimators]

    if len(cliques) == 0:
        warnings.warn(
            "No significant difference between the classifiers. "
            "Consider using critical difference diagrams.",
            stacklevel=2,
        )
    if reverse:
        ordered_labels = ordered_labels[::-1]
        ordered_avg_ranks = ordered_avg_ranks[::-1]
        ordered_avg_scores = ordered_avg_scores[::-1]
        for i in range(len(cliques)):
            cliques[i] = cliques[i][::-1]

    label_lengths = np.array([len(i) for i in ordered_labels])
    labels_space = 0.5 if max(label_lengths) > 14 else 0

    # Compress the graph by reducing vertical spacing
    spacing = 0.2 if len(cliques) >= 3 else 0.3

    height = (
        max(
            len(cliques) * 0.6 if len(cliques) > 5 else len(cliques),
            1.5 if n_estimators < 7 else 2.5,
        )
        + labels_space
    )
    width = max(n_estimators * 0.8, 7.5)

    fig, ax = plt.subplots(figsize=(width, height), layout="tight")

    # Draw the horizontal lines for each valid row
    for row_index, row in enumerate(cliques[::-1]):
        true_indices = np.where(row)[0]
        if true_indices.size > 0:
            y_pos = spacing * row_index
            for col_index in true_indices:
                # add circles for each 'True' value
                ax.plot(
                    col_index,
                    y_pos,
                    "o",
                    markersize=fontsize - 2,
                    color="black",
                )

            ax.hlines(
                y=y_pos,
                xmin=true_indices[-1],
                xmax=true_indices[0],
                color="black",
                linewidth=2,
            )

    # Setting labels for x-axis. Rotate only if labels are too long.
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(ordered_labels)))
    if (sum(label_lengths) > 40) or (max(label_lengths[:-1] + label_lengths[1:]) > 20):
        ax.set_xticklabels(ordered_labels, rotation=45, ha="left", fontsize=fontsize)
    else:
        ax.set_xticklabels(ordered_labels, fontsize=fontsize)

    ax.xaxis.set_label_position("top")

    y_pos = spacing * (len(cliques) + 2.5)
    # Add a horizontal line on the X axis under the labels
    ax.hlines(
        y=y_pos,
        xmin=-0.5,
        xmax=n_estimators - 0.5,
        color="black",
        linewidth=1,
    )

    for idx, (ran, val) in enumerate(zip(ordered_avg_ranks, ordered_avg_scores)):
        ax.text(
            idx,
            y_pos - spacing / 2,
            f"{ran:.3f}",
            ha="center",
            va="center",
            fontsize=fontsize,
        )

        ax.text(
            idx,
            y_pos - spacing * 1.5,
            f"{val:.2f}" if ordered_avg_scores.mean() > 9 else f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=fontsize - 2 if ordered_avg_scores.mean() > 9 else fontsize,
        )

    ax.text(
        -1,
        y_pos - spacing / 2,
        "Avg. rank",
        ha="right",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
    )

    ax.text(
        -1,
        y_pos - spacing * 1.5,
        "Avg. value",
        ha="right",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
    )

    # Adjust the y limits
    ax.set_ylim(-0.1, y_pos)
    ax.set_xlim(-1, n_estimators)

    # Remove the spines, ticks, labels, and grid
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False, labelbottom=False)
    ax.grid(False)

    if return_p_values:
        return fig, ax, p_values
    else:
        return fig, ax


# this _build_cliques is different to the CD one.
def _build_cliques(pairwise_matrix):
    """
    Build cliques from pairwise comparison matrix.

    Parameters
    ----------
    pairwise_matrix : np.array
        Pairwise matrix shape (n_estimators, n_estimators) indicating if there is a
        significant difference between pairs. Assumed to be ordered by rank of
        estimators.

    Returns
    -------
    list of lists
        cliques within which there is no significant different between estimators.
    """
    n = np.sum(pairwise_matrix, 1)
    cliques = pairwise_matrix[n > 1, :]

    for i in range(len(cliques) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if np.all(cliques[j, cliques[i, :]]):
                cliques[i, :] = 0
                break

    n = np.sum(cliques, 1)
    cliques = cliques[n > 1, :]

    return cliques
