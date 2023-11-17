# Adjusting the function to compress the graph further, reducing overall white space
import matplotlib.pyplot as plt
import numpy as np


def _plot_significance_groups(boolean_matrix, identifiers, names):
    """Plot significance relations.

    Plot a compressed graph with horizontal lines for each 'True' in a row of the
    boolean matrix, and a line connecting the 'True' elements in the row, with the X
    axis at the top, with names on the X axis, reduced overall white space, without
    grid lines, and with larger letters and circles. Rows that contain all false
    values are ignored.

    Parameters
    ----------
    boolean_matrix : A 2D array (n x n) of booleans
    identifiers: A list of strings of length n for graph labels
    names: A list of strings of length n for X axis labels
    """
    fig, ax = plt.subplots()
    num_cols = len(boolean_matrix[0])
    # Filter out the rows that contain all false values
    valid_rows = [row for row in boolean_matrix if np.sum(row) > 1]

    # Compress the graph by reducing vertical spacing
    total_height = len(valid_rows)
    spacing = (
        total_height / len(valid_rows) * 0.3 if len(valid_rows) > 0 else 0.3
    )  # Reducing the spacing

    # Draw the horizontal lines for each valid row
    for row_index, row in enumerate(valid_rows):
        true_indices = np.where(row)[0]
        if true_indices.size > 0:
            y_pos = spacing * (total_height - row_index - 1)  # Adjusting y position
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

    # Set the X axis labels to the provided names and move the X axis to the top
    ax.xaxis.tick_top()
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(reversed(names))  # Reversed names for the x-axis
    ax.xaxis.set_label_position("top")

    # Adjust the y limits to reduce overall white space
    ax.set_ylim(-0.5, spacing * total_height)

    # Remove the spines, ticks, labels, and grid
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False, labelbottom=False)
    ax.grid(False)

    plt.show()
