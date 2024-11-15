"""Function to compute and plot critical difference diagrams."""

__maintainer__ = []

__all__ = [
    "plot_critical_difference",
]

import math

import numpy as np
from scipy.stats import rankdata

from aeon.benchmarking.stats import check_friedman, nemenyi_test, wilcoxon_test
from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_critical_difference(
    scores,
    labels,
    highlight=None,
    lower_better=False,
    test="wilcoxon",
    correction="holm",
    alpha=0.1,
    width=6,
    textspace=1.5,
    reverse=True,
    return_p_values=False,
):
    """
    Plot the average ranks and cliques based on the method described in [1]_.

    This function summarises the relative performance of multiple estimators
    evaluated on multiple datasets. The resulting plot shows the average rank of each
    estimator on a number line. Estimators are grouped by solid lines,
    called cliques. A clique represents a group of estimators within which there is
    no significant difference in performance (see the caveats below). Please note
    that these diagrams are not an end in themselves, but rather form one part of
    the description of performance of estimators.

    The input is a summary performance measure of each estimator on each problem,
    where columns are estimators and rows datasets. This could be any measure such as
    accuracy/error, F1, negative log-likelihood, mean squared error or rand score.

    This algorithm first calculates the rank of all estimators on all problems (
    averaging ranks over ties), then sorts estimators on average rank. It then forms
    cliques. The original critical difference diagrams [1]_ use the post hoc Neymeni
    test [4]_ to find a critical difference. However, as discussed [3]_,this post hoc
    test is senstive to the estimators included in the test: "For instance the
    difference between A and B could be declared significant if the pool comprises
    algorithms C, D, E and not significant if the pool comprises algorithms
    F, G, H.". Our default option is to base cliques finding on pairwise Wilcoxon sign
    rank test.

    There are two issues when performing multiple pairwise tests to find cliques:
    what adjustment to make for multiple testing, and whether to perform a one sided
    or two sided test. The Bonferroni adjustment is known to be conservative. Hence,
    by default, we base our clique finding from pairwise tests on the control
    tests described in [1]_ and the sequential method recommended in [2]_ and proposed
    in [5]_ that uses a less conservative adjustment than Bonferroni.

    We perform all pairwise tests using a one-sided Wilcoxon sign rank test with the
    Holm correction to alpha, which involves reducing alpha by dividing it by number
    of estimators -1.

    Suppose we have four estimators, A, B, C and D sorted by average rank. Starting
    from A, we test the null hypothesis that average ranks are equal against the
    alternative hypothesis that the average rank of A is less than that of B,
    with significance level alpha/(n_estimators-1). If we reject the null hypothesis
    then we stop, and A is not in a clique. If we cannot
    reject the null, we test A vs C, continuing until we reject the null or we have
    tested all estimators. Suppose we find that A vs B is significant. We form no
    clique for A.

    We then continue to form a clique using the second best estimator,
    B, as a control. Imagine we find no difference between B and C, nor any difference
    between B and D. We have formed a clique for B: [B, C, D]. On the third
    iteration, imagine we also find not difference between C and D and thus form a
    second clique, [C, D]. We have found two cliques, but [C,D] is contained in [B, C,
    D] and is thus redundant. In this case we would return a single clique, [B, C, D].

    Not this is a heuristic approach not without problems: If the best ranked estimator
    A is significantly better than B but not significantly different to C, this will
    not be reflected in the diagram. Because of this, we recommend also reporting
    p-values in a table, and exploring other ways to present results such as pairwise
    plots. Comparing estimators on archive data sets can only be indicative of
    overall performance in general, and such comparisons should be seen as exploratory
    analysis rather than designed experiments to test an a priori hypothesis.

    Parts of the code are adapted from here:
    https://github.com/hfawaz/cd-diagram

    Parameters
    ----------
    scores : np.array
        Performance scores for estimators of shape (n_datasets, n_estimators).
    labels : list of estimators
        List with names of the estimators. Order should be the same as scores
    highlight : dict, default = None
        A dict with labels and HTML colours to be used for highlighting. Order should be
        the same as scores.
    lower_better : bool, default = False
        Indicates whether smaller is better for the results in scores. For example,
        if errors are passed instead of accuracies, set ``lower_better`` to ``True``.
    test : string, default = "wilcoxon"
        test method used to form cliques, either "nemenyi" or "wilcoxon"
    correction: string, default = "holm"
        correction method for multiple testing, one of "bonferroni", "holm" or "none".
    alpha : float, default = 0.1
        Critical value for statistical tests of difference.
    width : int, default = 6
        Width in inches.
    textspace : int
        Space on figure sides (in inches) for the method names (default: 1.5).
    reverse : bool, default = True
        If set to 'True', the lowest rank is on the right.
    return_p_values : bool, default = False
        Whether to return the pairwise matrix of p-values.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    p_values : np.ndarray (optional)
        if return_p_values is True, returns a (n_estimators, n_estimators) matrix of
        unadjusted p values for the pairwise Wilcoxon sign rank test.

    References
    ----------
    .. [1] Demsar J., "Statistical comparisons of classifiers over multiple data sets."
    Journal of Machine Learning Research 7:1-30, 2006.
    .. [2] García S. and Herrera F., "An extension on “statistical comparisons of
    classifiers over multiple data sets” for all pairwise comparisons."
    Journal of Machine Learning Research 9:2677-2694, 2008.
    .. [3] Benavoli A., Corani G. and Mangili F "Should we really use post-hoc tests
    based on mean-ranks?" Journal of Machine Learning Research 17:1-10, 2016.
    .. [4] Nemenyi P., "Distribution-free multiple comparisons".
    PhD thesis, Princeton University, 1963.
    .. [5] Holm S., " A simple sequentially rejective multiple test procedure."
    Scandinavian Journal of Statistics, 6:65-70, 1979.

    Examples
    --------
    >>> from aeon.visualisation import plot_critical_difference
    >>> from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    >>> methods = ["IT", "WEASEL-Dilation", "HIVECOTE2", "FreshPRINCE"]
    >>> results = get_estimator_results_as_array(estimators=methods) # doctest: +SKIP
    >>> plot = plot_critical_difference(results[0], methods, alpha=0.1)\
        # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    >>> plot.savefig("cd.pdf", bbox_inches="tight")  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    if isinstance(scores, list):
        scores = np.array(scores)

    n_datasets, n_estimators = scores.shape
    if isinstance(test, str):
        test = test.lower()
    if isinstance(correction, str):
        correction = correction.lower()

    p_values = None
    if return_p_values and test != "wilcoxon":
        raise ValueError(
            f"Cannot return p values for the {test}, since it does "
            "not calculate p-values."
        )

    # Step 1: rank data: in case of ties average ranks are assigned
    if lower_better:  # low is good -> rank 1
        ranks = rankdata(scores, axis=1)
    else:  # assign opposite ranks
        ranks = rankdata(-1 * scores, axis=1)

    # Step 2: calculate average rank per estimator
    ordered_avg_ranks = ranks.mean(axis=0)
    # Sort labels and ranks
    ordered_labels_ranks = np.array(
        [(labels, float(r)) for r, labels in sorted(zip(ordered_avg_ranks, labels))],
        dtype=object,
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
    p_value_friedman = check_friedman(ranks)
    # Step 4: If Friedman test is significant find cliques
    if p_value_friedman < alpha:
        if test == "nemenyi":
            cliques = nemenyi_test(ordered_avg_ranks, n_datasets, alpha)
            cliques = _build_cliques(cliques)
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
        """List location in list of list structure."""
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
    linewidth = 0.75
    linewidth_sign = 2.5

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
            size=11,
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

    if return_p_values:
        return fig, ax, p_values
    else:
        return fig, ax


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
    for i in range(0, pairwise_matrix.shape[0]):
        for j in range(i + 1, pairwise_matrix.shape[1]):
            if pairwise_matrix[i, j] == 0:
                pairwise_matrix[i, j + 1 :] = 0  # noqa: E203
                break

    n = np.sum(pairwise_matrix, 1)
    possible_cliques = pairwise_matrix[n > 1, :]

    for i in range(possible_cliques.shape[0] - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if np.all(possible_cliques[j, possible_cliques[i, :]]):
                possible_cliques[i, :] = 0
                break

    n = np.sum(possible_cliques, 1)
    cliques = possible_cliques[n > 1, :]

    return cliques
