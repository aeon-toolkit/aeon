import numpy as np
from scipy.stats import rankdata

from aeon.performance_metrics.stats import check_friedman, nemenyi_test, wilcoxon_test
from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_significance(
    scores,
    labels,
    alpha=0.1,
    lower_better=False,
    test="wilcoxon",
    correction="holm",
):
    """
    Plot that allow the case where cliques can be deceiving.

    Plot a graph with a horizontal line on the X axis under the labels, with horizontal
    lines for each row of the boolean matrix that has more than one 'True' value, and a
    line connecting the 'True' elements in those rows, with the X axis at the top,
    with labels on the X axis, reduced overall white space, without grid lines,
    and with larger letters and circles.

    Parameters
    ----------
    - boolean_matrix: A 2D numpy array (n x n) of booleans
    - labels: A list of strings of length n for X axis labels
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

    # Step 2: calculate average rank per estimator
    ordered_avg_ranks = ranks.mean(axis=0)
    # Sort labels and ranks
    ordered_labels_ranks = np.array(
        [(l, float(r)) for r, l in sorted(zip(ordered_avg_ranks, labels))], dtype=object
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

    # Compress the graph by reducing vertical spacing
    spacing = 0.3

    height = len(cliques) * 1

    width = float(len(labels))

    fig, ax = plt.subplots(figsize=(width, height))

    num_rows, num_cols = cliques.shape

    # Draw the horizontal lines for each valid row
    for row_index, row in enumerate(cliques):
        true_indices = np.where(row)[0]
        if true_indices.size > 0:
            y_pos = spacing * row_index
            for col_index in true_indices:
                # add circles for each 'True' value
                ax.plot(
                    num_cols - col_index - 1,
                    y_pos,
                    "o",
                    markersize=13,
                    color="black",
                )

            ax.hlines(
                y=y_pos,
                xmin=num_cols - true_indices[-1] - 1,
                xmax=num_cols - true_indices[0] - 1,
                color="black",
                linewidth=2,
            )

    # Setting labels for x-axis. Rotate only if labels are too long.
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(ordered_labels)))
    label_lengths = np.array([len(i) for i in reversed(ordered_labels)])
    if (sum(label_lengths) > 40) or (max(label_lengths[:-1] + label_lengths[1:]) > 20):
        ax.set_xticklabels(
            reversed(ordered_labels), rotation=45, ha="left", fontsize=14
        )
    else:
        ax.set_xticklabels(reversed(ordered_labels), fontsize=14)

    ax.xaxis.set_label_position("top")

    y_pos = y_pos + spacing * 2

    # Add a horizontal line on the X axis under the labels
    ax.hlines(
        y=y_pos,
        xmin=-0.5,
        xmax=num_cols - 0.5,
        color="black",
        linewidth=1,
    )
    ordered_avg_ranks = list(reversed(ordered_avg_ranks))
    # add ranks below the horizontal line

    ordered_avg_scores = list(reversed(ordered_avg_scores))
    for i in range(len(ordered_avg_ranks)):
        ax.text(
            i,
            y_pos - 0.1,
            f"{ordered_avg_ranks[i]:.3f}",
            ha="center",
            va="center",
            fontsize=14,
        )

        ax.text(
            i,
            y_pos - 0.3,
            f"{ordered_avg_scores[i]:.3f}",
            ha="center",
            va="center",
            fontsize=14,
        )

    ax.text(
        -1,
        y_pos - 0.1,
        "Avg. rank",
        ha="right",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        -1,
        y_pos - 0.3,
        "Avg. value",
        ha="right",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust the y limits
    ax.set_ylim(-0.1, y_pos)

    # Remove the spines, ticks, labels, and grid
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False, labelbottom=False)
    ax.grid(False)

    plt.tight_layout()

    return fig


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

    return cliques[::-1]
