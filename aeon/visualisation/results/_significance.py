import numpy as np


def plot_significance(boolean_matrix, identifiers, names):
    """
    Plot that allow the case where cliques can be deceiving.

    Plot a graph with a horizontal line on the X axis under the labels, with horizontal
    lines for each row of the boolean matrix that has more than one 'True' value, and a
    line connecting the 'True' elements in those rows, with the X axis at the top,
    with names on the X axis, reduced overall white space, without grid lines,
    and with larger letters and circles.

    Parameters
    ----------
    - boolean_matrix: A 2D numpy array (n x n) of booleans
    - identifiers: A list of strings of length n for graph labels
    - names: A list of strings of length n for X axis labels
    """
    import matplotlib.pyplot as plt

    # Filter out the rows that have more than one 'True' value
    valid_rows = _build_cliques(boolean_matrix)

    # Compress the graph by reducing vertical spacing
    spacing = 0.4

    height = len(identifiers) * spacing

    width = float(len(identifiers))

    fig, ax = plt.subplots(figsize=(width, height))

    num_rows, num_cols = boolean_matrix.shape

    # Draw the horizontal lines for each valid row
    for row_index, row in enumerate(valid_rows):
        true_indices = np.where(row)[0]
        if true_indices.size > 0:
            y_pos = spacing * row_index
            for col_index in true_indices:
                ax.text(
                    num_cols - col_index - 1,
                    y_pos,
                    identifiers[col_index],
                    ha="center",
                    va="center",
                    fontsize=16,
                    bbox=dict(
                        facecolor="white", edgecolor="black", boxstyle="circle,pad=0.5"
                    ),
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
    ax.set_xticks(np.arange(len(names)))
    label_lengths = np.array([len(i) for i in reversed(names)])
    if (sum(label_lengths) > 40) or (max(label_lengths[:-1] + label_lengths[1:]) > 20):
        ax.set_xticklabels(reversed(names), rotation=45, ha="left")
    else:
        ax.set_xticklabels(reversed(names))

    ax.xaxis.set_label_position("top")

    y_pos = y_pos + spacing

    # Add a horizontal line on the X axis under the labels
    ax.hlines(
        y=y_pos,
        xmin=-0.5,
        xmax=num_cols - 0.5,
        color="black",
        linewidth=1,
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
    possible_cliques = pairwise_matrix[n > 1, :]

    for i in range(len(possible_cliques) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if np.all(possible_cliques[j, possible_cliques[i, :]]):
                possible_cliques[i, :] = 0
                break

    n = np.sum(possible_cliques, 1)
    cliques = possible_cliques[n > 1, :]

    return cliques
