"""Function to compute and plot critical difference diagrams."""

__author__ = ["SveaMeyer13", "dguijo"]

import math

import numpy as np

# import pandas as pd
from scipy.stats import distributions, find_repeats, rankdata, wilcoxon

from aeon.benchmarking.utils import get_qalpha
from aeon.utils.validation._dependencies import _check_soft_dependencies


def _check_friedman(ranks):
    """
    Check whether Friedman test is significant.

    Larger parts of code copied from scipy.

    Parameters
    ----------
    n_estimators : int
      number of strategies to evaluate
    n_datasets : int
      number of datasets classified per strategy
    ranks : np.array (shape: n_estimators * n_datasets)
      rank of strategy on dataset

    Returns
    -------
    is_significant : bool
      Indicates whether strategies differ significantly in terms of performance
      (according to Friedman test).
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
    return chisq, p_value


def nemenyi_test(n_estimators, n_datasets, ordered_avg_ranks, alpha):
    """Find cliques using post hoc Nemenyi test."""
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

    cliques = _build_cliques(cliques)

    return cliques


def wilcoxon_test(results, adjusted_alpha):
    """
    Perform Wilcoxon test.

    Parameters
    ----------
    results: np.array
      results of strategies on datasets
    adjusted_alpha: float
        alpha level adjusted for multiple testing

    Returns
    -------
    cliques: list of lists
        statistically similar cliques
    p_values: np.array
        p-values of Wilcoxon test
    """
    n_estimators = results.shape[1]

    p_values = np.eye(n_estimators)

    for i in range(n_estimators - 1):
        for j in range(i + 1, n_estimators):
            p_values[i, j] = wilcoxon(
                results[:, i], results[:, j], zero_method="wilcox"
            )[1]

    cliques = _build_cliques(p_values > adjusted_alpha)

    return cliques, p_values


def _build_cliques(pairwise_matrix):
    """
    Build cliques from pairwise comparison matrix.

    Parameters
    ----------
    pairwise_matrix: np.array
        pairwise comparison matrix

    Returns
    -------
    cliques: list of lists
        statistically similar cliques
    """
    n = np.sum(pairwise_matrix, 1)
    possible_cliques = pairwise_matrix[n > 1, :]

    # all False values between two True values are set to True as they have to be
    # represented as significant differences
    for i in range(0, possible_cliques.shape[0]):
        true_limits = np.where(possible_cliques[i, :] == 1)[0]
        possible_cliques[i, true_limits[0] : true_limits[-1]] = 1  # noqa: E203

    for i in range(possible_cliques.shape[0] - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if np.all(possible_cliques[j, possible_cliques[i, :]]):
                possible_cliques[i, :] = 0
                break

    n = np.sum(possible_cliques, 1)
    cliques = possible_cliques[n > 1, :]

    return cliques


def plot_critical_difference(
    scores,
    labels,
    highlight=None,
    errors=False,
    test="wilconxon",
    correction="holm",
    alpha=0.05,
    width=6,
    textspace=1.5,
    reverse=True,
):
    """
    Find cliques using Wilcoxon and post hoc Holm test.

    Computes groups of estimators (cliques) within which there is no significant
    difference. The algorithm assumes that the estimators (named in labels) are
    sorted by average rank, so that labels[0] is the "best" estimator in terms of
    lowest average rank.

    This algorithm first forms a clique for each estimator set as control, then merges
    dominated cliques.

    Suppose we have four estimators, A, B, C and D sorted by average rank. Starting
    from A, we test the null hypothesis that average ranks are equal against the
    alternative hypothesis that the average rank of A is less than that of B. If we
    reject the null hypothesis then we stop, and A is not in a clique. If we cannot
    reject the null, we test A vs C, continuing until we reject the null or we have
    tested all estimators.

    Suppose we find B is significantly worse that A, but that on the next iteration we
    find no difference between B and C, nor any difference between B and D. We have
    formed one clique, [B, C, D]. On the third iteration, we also find not difference
    between C and D and thus form a second clique, [C, D]. We have found two cliques,
    but [C,D] is contained in [B, C, D] and is thus redundant. In this case we would
    return a single clique, [ B, C, D].

    Parts of the code are copied and adapted from here:
    https://github.com/hfawaz/cd-diagram

    Parameters
    ----------
        scores : np.array
            scores (either accuracies or errors) of dataset x strategy
        labels : list of estimators
            list with names of the estimators. Order should be the same as scores
        highlight: dict with labels and HTML colours to be used, default = None
            dict with labels and HTML colours to be used for highlighting. Order should
            be the same as scores
        errors : bool, default = False
            indicates whether scores are passed as errors (default) or accuracies
        test : string, default = "wilcoxon"
            test method, to include "nemenyi" and "wilcoxon"
        correction: string, default = "holm"
            correction method, to include "bonferroni", "holm" and None
        alpha : float default = 0.05
             Alpha level for statistical tests currently supported: 0.1, 0.05 or 0.01)
        width : int, default = 6
           width in inches
        textspace : int
           space on figure sides (in inches) for the method names (default: 1.5)
        reverse : bool, default = True
           if set to 'True', the lowest rank is on the right

    Returns
    -------
    fig: matplotlib.figure
        Figure created.

    Example
    -------
    >>> from aeon.benchmarking import plot_critical_difference
    >>> from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    >>> methods = ["IT", "WEASEL-Dilation", "HIVECOTE2", "FreshPRINCE"]
    >>> results = get_estimator_results_as_array(estimators=methods)
    >>> plot = plot_critical_difference(results[0], methods, alpha=0.1)\
        # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    >>> plot.savefig("cd.pdf", bbox_inches="tight")  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    # Helper Functions
    # get number of datasets and strategies:
    n_datasets, n_estimators = scores.shape

    # Step 1: rank data: best algorithm gets rank of 1 second best rank of 2...
    # in case of ties average ranks are assigned
    if errors:
        # low is good -> rank 1
        ranks = rankdata(scores, axis=1)
    else:
        # assign opposite ranks
        ranks = rankdata(-1 * scores, axis=1)

    # Step 2: calculate average rank per strategy
    ordered_avg_ranks = ranks.mean(axis=0)
    # Sort labels
    ordered_labels_ranks = np.array(
        [(l, float(r)) for r, l in sorted(zip(ordered_avg_ranks, labels))], dtype=object
    )
    ordered_labels = np.array([la for la, _ in ordered_labels_ranks], dtype=str)
    ordered_avg_ranks = np.array([r for _, r in ordered_labels_ranks], dtype=np.float32)

    indices = [np.where(np.array(labels) == r)[0] for r in ordered_labels]

    ordered_scores = scores[:, indices]
    # sort out colours for labels
    if highlight is not None:
        colours = [
            highlight[label] if label in highlight else "#000000"
            for label in ordered_labels
        ]
    else:
        colours = ["#000000"] * len(ordered_labels)

    # Step 3 : check whether Friedman test is significant
    _, p_value_friedman = _check_friedman(ranks)
    # Step 4: If Friedman test is significant find cliques
    if p_value_friedman < alpha:
        if test == "nemenyi":
            cliques = nemenyi_test(n_estimators, n_datasets, ordered_avg_ranks, alpha)
        elif test == "wilcoxon":
            if correction == "bonferroni":
                adjusted_alpha = alpha / (n_estimators * (n_estimators - 1) / 2)
            elif correction == "holm":
                adjusted_alpha = alpha / (n_estimators - 1)
            elif correction is None:
                adjusted_alpha = alpha
            else:
                raise ValueError("correction available are None, bonferroni and holm.")
            cliques, p_values = wilcoxon_test(ordered_scores, adjusted_alpha)
        else:
            raise ValueError("tests available are only nemenyi and wilcoxon.")
    # If Friedman test is not significant everything has to be one clique
    else:
        cliques = [[1] * n_estimators]

    # pd.DataFrame(p_values, index=ordered_labels, columns=ordered_labels).round(
    #     6
    # ).to_csv("./p_values.csv")

    # print(adjusted_alpha)

    # pd.DataFrame(
    #     p_values > adjusted_alpha,
    #     index=ordered_labels,
    #     columns=ordered_labels,
    # ).to_csv("./same.csv")

    # Step 6 create the diagram:
    # check from where to where the axis has to go
    lowv = min(1, int(math.floor(min(ordered_avg_ranks))))
    highv = max(len(ordered_avg_ranks), int(math.ceil(max(ordered_avg_ranks))))

    # set up the figure
    width = float(width)
    textspace = float(textspace)

    cline = 0.6  # space needed above scale
    linesblank = 1  # lines between scale and text
    scalewidth = width - 2 * textspace

    # calculate needed height
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((n_estimators + 1) / 2) * 0.2 + minnotsignificant + 0.2

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(1, 0)

    def _lloc(lst, n):
        """
        List location in list of list structure.

        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(lst[0]) + n
        else:
            return n

    def _nth(lst, n):
        n = _lloc(lst, n)
        return [a[n] for a in lst]

    def _hfl(lst):
        return [a * hf for a in lst]

    def _wfl(lst):
        return [a * wf for a in lst]

    def _line(lst, color="k", **kwargs):
        ax.plot(_wfl(_nth(lst, 0)), _hfl(_nth(lst, 1)), color=color, **kwargs)

    # draw scale
    _line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    def _rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    # add ticks to scale
    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        _line([(_rankpos(a), cline - tick / 2), (_rankpos(a), cline)], linewidth=2)

    def _text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    for a in range(lowv, highv + 1):
        _text(
            _rankpos(a),
            cline - tick / 2 - 0.05,
            str(a),
            ha="center",
            va="bottom",
            size=16,
        )

    # sort out lines and text based on whether order is reversed or not
    space_between_names = 0.24
    for i in range(math.ceil(len(ordered_avg_ranks) / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        if reverse:
            _line(
                [
                    (_rankpos(ordered_avg_ranks[i]), cline),
                    (_rankpos(ordered_avg_ranks[i]), chei),
                    (textspace + scalewidth + 0.2, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels left side.
                textspace + scalewidth + 0.3,
                chei,
                ordered_labels[i],
                ha="left",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks left side.
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(ordered_avg_ranks[i], ".4f"),
                ha="left",
                va="center",
                size=10,
                color=colours[i],
            )
        else:
            _line(
                [
                    (_rankpos(ordered_avg_ranks[i]), cline),
                    (_rankpos(ordered_avg_ranks[i]), chei),
                    (textspace - 0.1, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels left side.
                textspace - 0.2,
                chei,
                ordered_labels[i],
                ha="right",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks left side.
                textspace + 0.4,
                chei - 0.075,
                format(ordered_avg_ranks[i], ".4f"),
                ha="right",
                va="center",
                size=10,
                color=colours[i],
            )

    for i in range(math.ceil(len(ordered_avg_ranks) / 2), len(ordered_avg_ranks)):
        chei = (
            cline
            + minnotsignificant
            + (len(ordered_avg_ranks) - i - 1) * space_between_names
        )
        if reverse:
            _line(
                [
                    (_rankpos(ordered_avg_ranks[i]), cline),
                    (_rankpos(ordered_avg_ranks[i]), chei),
                    (textspace - 0.1, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels right side.
                textspace - 0.2,
                chei,
                ordered_labels[i],
                ha="right",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks right side.
                textspace + 0.4,
                chei - 0.075,
                format(ordered_avg_ranks[i], ".4f"),
                ha="right",
                va="center",
                size=10,
                color=colours[i],
            )
        else:
            _line(
                [
                    (_rankpos(ordered_avg_ranks[i]), cline),
                    (_rankpos(ordered_avg_ranks[i]), chei),
                    (textspace + scalewidth + 0.1, chei),
                ],
                linewidth=linewidth,
                color=colours[i],
            )
            _text(  # labels right side.
                textspace + scalewidth + 0.2,
                chei,
                ordered_labels[i],
                ha="left",
                va="center",
                size=16,
                color=colours[i],
            )
            _text(  # ranks right side.
                textspace + scalewidth - 0.4,
                chei - 0.075,
                format(ordered_avg_ranks[i], ".4f"),
                ha="left",
                va="center",
                size=10,
                color=colours[i],
            )

    # draw lines for cliques
    start = cline + 0.2
    side = -0.02 if reverse else 0.02
    height = 0.1
    i = 1
    for clq in cliques:
        positions = np.where(np.array(clq) == 1)[0]
        min_idx = np.array(positions).min()
        max_idx = np.array(positions).max()
        _line(
            [
                (_rankpos(ordered_avg_ranks[min_idx]) - side, start),
                (_rankpos(ordered_avg_ranks[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
        )
        start += height

    return fig
