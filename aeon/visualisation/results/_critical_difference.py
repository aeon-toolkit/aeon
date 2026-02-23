"""Function to compute and plot critical difference diagrams."""

__maintainer__ = ["baraline"]

__all__ = [
    "plot_critical_difference",
]

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from aeon.benchmarking.stats import check_friedman, nemenyi_test, wilcoxon_test
from aeon.utils.validation._dependencies import _check_soft_dependencies

# =============================================================================
# Constants
# =============================================================================

# Scale drawing constants
# -----------------------
# Width of the main horizontal scale line in points
_SCALE_LINEWIDTH = 2
# Height of tick marks at integer rank positions (in figure units/inches)
_MAJOR_TICK_HEIGHT = 0.3
# Height of tick marks at half-integer rank positions (in figure units/inches)
_MINOR_TICK_HEIGHT = 0.15
# Font size for the numeric rank labels on the scale
_SCALE_LABEL_SIZE = 11

# Estimator drawing constants
# ---------------------------
# Width of the L-shaped lines connecting estimators to the scale
_ESTIMATOR_LINEWIDTH = 0.75
# Vertical spacing between consecutive estimator lines (in figure units/inches)
_ESTIMATOR_SPACING = 0.24
# Font size for the average rank value displayed above each estimator line
_RANK_TEXT_SIZE = 10
# Font size for the estimator name labels
_LABEL_TEXT_SIZE = 16
# Horizontal gap between the line endpoint and the estimator label (in figure units)
_LABEL_OFFSET = 0.1

# Clique drawing constants
# ------------------------
# Width of the horizontal bars connecting estimators in the same clique
_CLIQUE_LINEWIDTH = 2.5
# Vertical spacing between multiple clique bars (in figure units/inches)
_CLIQUE_SPACING = 0.1
# Vertical offset from the scale to the first clique bar (in figure units/inches)
_CLIQUE_START_OFFSET = 0.2
# Small horizontal extension of clique bars beyond the outermost estimator positions
_CLIQUE_SIDE_OFFSET = 0.02

# Figure layout constants
# -----------------------
# Y-position of the main scale line from the top of the figure (in figure units)
_CLINE_POSITION = 0.6
# Minimum number of blank line heights below the scale before estimator lines start
_MIN_LINES_BLANK = 1


# =============================================================================
# Rank Computation Functions
# =============================================================================


def _compute_ranks(scores: np.ndarray, lower_better: bool) -> np.ndarray:
    """
    Compute ranks for each estimator on each dataset.

    Ranks are computed row-wise (per dataset). In case of ties, average ranks
    are assigned.

    Parameters
    ----------
    scores : np.ndarray of shape (n_datasets, n_estimators)
        Performance scores for each estimator on each dataset.
    lower_better : bool
        If True, lower scores receive better (lower) ranks.
        If False, higher scores receive better ranks.

    Returns
    -------
    ranks : np.ndarray of shape (n_datasets, n_estimators)
        Ranks for each estimator on each dataset, where rank 1 is best.
    """
    if lower_better:
        return rankdata(scores, axis=1)
    return rankdata(-1 * scores, axis=1)


def _sort_by_average_rank(
    scores: np.ndarray, labels: list[str], ranks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort estimators by their average rank across all datasets.

    Parameters
    ----------
    scores : np.ndarray of shape (n_datasets, n_estimators)
        Performance scores for each estimator on each dataset.
    labels : list of str
        Names of the estimators.
    ranks : np.ndarray of shape (n_datasets, n_estimators)
        Ranks for each estimator on each dataset.

    Returns
    -------
    ordered_labels : np.ndarray of str
        Estimator labels sorted by average rank (best first).
    ordered_avg_ranks : np.ndarray of float32
        Average ranks in ascending order.
    ordered_scores : np.ndarray
        Scores reordered to match sorted labels.
    """
    avg_ranks = ranks.mean(axis=0)
    sort_indices = np.argsort(avg_ranks)

    labels_array = np.array(labels, dtype=str)
    ordered_labels = labels_array[sort_indices]
    ordered_avg_ranks = avg_ranks[sort_indices].astype(np.float32)
    ordered_scores = scores[:, sort_indices]

    return ordered_labels, ordered_avg_ranks, ordered_scores


# =============================================================================
# Statistical Testing Functions
# =============================================================================


def _compute_adjusted_alpha(
    alpha: float, n_estimators: int, correction: str | None
) -> float:
    """
    Compute the adjusted significance level for multiple testing correction.

    Parameters
    ----------
    alpha : float
        Original significance level.
    n_estimators : int
        Number of estimators being compared.
    correction : str or None
        Correction method: "bonferroni", "holm", or None.

    Returns
    -------
    adjusted_alpha : float
        Adjusted significance level.

    Raises
    ------
    ValueError
        If correction method is not recognized.
    """
    if correction == "bonferroni":
        return alpha / (n_estimators * (n_estimators - 1) / 2)
    elif correction == "holm":
        return alpha / (n_estimators - 1)
    elif correction is None:
        return alpha
    else:
        raise ValueError("correction available are None, Bonferroni and Holm.")


def _find_cliques(
    ranks: np.ndarray,
    ordered_scores: np.ndarray,
    ordered_labels: np.ndarray,
    ordered_avg_ranks: np.ndarray,
    n_datasets: int,
    n_estimators: int,
    test: str,
    correction: str | None,
    alpha: float,
    lower_better: bool,
) -> tuple[list[list[int]], np.ndarray | None]:
    """
    Find groups (cliques) of estimators with no significant performance difference.

    First performs a Friedman test to check if there are any significant differences
    among the estimators. If the Friedman test is not significant, all estimators
    form a single clique. Otherwise, uses either the Nemenyi or Wilcoxon test
    to determine which pairs of estimators are significantly different.

    Parameters
    ----------
    ranks : np.ndarray of shape (n_datasets, n_estimators)
        Ranks for each estimator on each dataset.
    ordered_scores : np.ndarray
        Scores ordered by average rank.
    ordered_labels : np.ndarray
        Labels ordered by average rank.
    ordered_avg_ranks : np.ndarray
        Average ranks in ascending order.
    n_datasets : int
        Number of datasets.
    n_estimators : int
        Number of estimators.
    test : str
        Statistical test: "nemenyi" or "wilcoxon".
    correction : str or None
        Multiple testing correction method for Wilcoxon test.
    alpha : float
        Significance level.
    lower_better : bool
        Whether lower scores are better.

    Returns
    -------
    cliques : list of list
        Each inner list is a binary mask indicating clique membership.
    p_values : np.ndarray or None
        P-values matrix from Wilcoxon test, or None if Nemenyi test was used.

    Raises
    ------
    ValueError
        If test method is not recognized.
    """
    p_values = None
    p_value_friedman = check_friedman(ranks)

    # If Friedman test is not significant, all estimators form one clique
    if p_value_friedman >= alpha:
        p_values = np.triu(np.ones((n_estimators, n_estimators)))
        return [[1] * n_estimators], p_values

    # Perform post-hoc test
    if test == "nemenyi":
        cliques = nemenyi_test(ordered_avg_ranks, n_datasets, alpha)
        return _build_cliques(cliques), None

    if test == "wilcoxon":
        adjusted_alpha = _compute_adjusted_alpha(alpha, n_estimators, correction)
        p_values = wilcoxon_test(ordered_scores, ordered_labels, lower_better)
        return _build_cliques(p_values > adjusted_alpha), p_values

    raise ValueError("tests available are only nemenyi and wilcoxon.")


def _build_cliques(pairwise_matrix: np.ndarray) -> np.ndarray:
    """
    Build non-redundant cliques from a pairwise comparison matrix.

    A clique is a group of estimators where no pair has a significant difference.
    This function identifies all maximal cliques and removes redundant ones
    (cliques that are subsets of larger cliques).

    Parameters
    ----------
    pairwise_matrix : np.ndarray of shape (n_estimators, n_estimators)
        Upper triangular matrix where entry (i,j) is True/1 if estimators i and j
        are NOT significantly different. Assumed to be ordered by rank.

    Returns
    -------
    cliques : np.ndarray
        Array where each row is a binary mask indicating clique membership.
        Returns empty array if no cliques with more than one member exist.
    """
    # Propagate non-significance: if i and j are different, i and k (k>j) are different
    for i in range(pairwise_matrix.shape[0]):
        for j in range(i + 1, pairwise_matrix.shape[1]):
            if pairwise_matrix[i, j] == 0:
                pairwise_matrix[i, j + 1 :] = 0
                break

    # Keep only rows that could form cliques (connected to more than just themselves)
    n = np.sum(pairwise_matrix, 1)
    possible_cliques = pairwise_matrix[n > 1, :]

    # Remove redundant cliques (those contained in larger cliques)
    for i in range(possible_cliques.shape[0] - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if np.all(possible_cliques[j, possible_cliques[i, :].astype(bool)]):
                possible_cliques[i, :] = 0
                break

    # Return only non-empty cliques
    n = np.sum(possible_cliques, 1)
    return possible_cliques[n > 1, :]


# =============================================================================
# Figure Setup Functions
# =============================================================================


def _compute_figure_dimensions(
    n_estimators: int, width: float, textspace: float
) -> tuple[float, float, float, float]:
    """
    Compute the dimensions and layout parameters for the figure.

    Parameters
    ----------
    n_estimators : int
        Number of estimators to display.
    width : float
        Figure width in inches.
    textspace : float
        Space reserved on each side for labels.

    Returns
    -------
    height : float
        Figure height in inches.
    scale_y : float
        Y-position of the horizontal scale line.
    scalewidth : float
        Width of the scale (figure width minus label spaces).
    min_y_offset : float
        Minimum vertical space below scale for estimator lines.
    """
    scale_y = _CLINE_POSITION
    scalewidth = width - 2 * textspace
    min_y_offset = max(2 * _ESTIMATOR_SPACING, _MIN_LINES_BLANK)
    height = (
        scale_y + ((n_estimators + 1) / 2) * _ESTIMATOR_SPACING + min_y_offset + 0.2
    )

    return height, scale_y, scalewidth, min_y_offset


def _create_figure(width: float, height: float) -> tuple["Figure", "Axes"]:
    """
    Create and configure the matplotlib figure and axes.

    Parameters
    ----------
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes for drawing.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Initialize axes with inverted y-axis (0 at top)
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(1, 0)

    return fig, ax


def _draw_line(
    ax: "Axes",
    width: float,
    height: float,
    points: list[tuple[float, float]],
    color: str = "k",
    **kwargs,
) -> None:
    """
    Draw a line through the given points in figure coordinates.

    Converts logical coordinates (in inches/figure units) to normalized
    axes coordinates before drawing.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    width : float
        Figure width for coordinate conversion.
    height : float
        Figure height for coordinate conversion.
    points : list of tuple
        List of (x, y) coordinates in figure units.
    color : str, default="k"
        Line color.
    **kwargs
        Additional arguments passed to ax.plot().
    """
    wf = 1.0 / width
    hf = 1.0 / height
    xs = [p[0] * wf for p in points]
    ys = [p[1] * hf for p in points]
    ax.plot(xs, ys, color=color, **kwargs)


def _draw_text(
    ax: "Axes",
    width: float,
    height: float,
    x: float,
    y: float,
    s: str,
    **kwargs,
) -> None:
    """
    Draw text at the given position in figure coordinates.

    Converts logical coordinates (in inches/figure units) to normalized
    axes coordinates before drawing.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    width : float
        Figure width for coordinate conversion.
    height : float
        Figure height for coordinate conversion.
    x : float
        X-coordinate in figure units.
    y : float
        Y-coordinate in figure units.
    s : str
        Text to draw.
    **kwargs
        Additional arguments passed to ax.text().
    """
    wf = 1.0 / width
    hf = 1.0 / height
    ax.text(wf * x, hf * y, s, **kwargs)


def _rank_to_position(
    rank: float,
    textspace: float,
    scalewidth: float,
    min_rank: int,
    max_rank: int,
    reverse: bool,
) -> float:
    """
    Convert a rank value to its x-coordinate on the scale.

    Parameters
    ----------
    rank : float
        The rank value to convert.
    textspace : float
        Space reserved on each side for labels.
    scalewidth : float
        Width of the scale.
    min_rank : int
        Lowest rank value on the scale.
    max_rank : int
        Highest rank value on the scale.
    reverse : bool
        If True, lower ranks appear on the right side.

    Returns
    -------
    x : float
        The x-coordinate corresponding to the rank.
    """
    if reverse:
        normalized = max_rank - rank
    else:
        normalized = rank - min_rank
    return textspace + scalewidth / (max_rank - min_rank) * normalized


# =============================================================================
# Drawing Functions
# =============================================================================


def _draw_scale(
    ax: "Axes",
    width: float,
    height: float,
    textspace: float,
    scalewidth: float,
    scale_y: float,
    min_rank: int,
    max_rank: int,
    reverse: bool,
) -> None:
    """
    Draw the horizontal scale with tick marks and numeric labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    width : float
        Figure width.
    height : float
        Figure height.
    textspace : float
        Space reserved for labels.
    scalewidth : float
        Width of the scale.
    scale_y : float
        Y-position of the scale line.
    min_rank : int
        Lowest rank value.
    max_rank : int
        Highest rank value.
    reverse : bool
        If True, lower ranks appear on the right side.
    """
    # Draw main horizontal scale line
    _draw_line(
        ax,
        width,
        height,
        [(textspace, scale_y), (width - textspace, scale_y)],
        linewidth=_SCALE_LINEWIDTH,
    )

    # Draw tick marks at half-integer intervals
    for tick_value in list(np.arange(min_rank, max_rank, 0.5)) + [max_rank]:
        is_major = tick_value == int(tick_value)
        tick_height = _MAJOR_TICK_HEIGHT if is_major else _MINOR_TICK_HEIGHT
        x = _rank_to_position(
            tick_value, textspace, scalewidth, min_rank, max_rank, reverse
        )
        _draw_line(
            ax,
            width,
            height,
            [(x, scale_y - tick_height / 2), (x, scale_y)],
            linewidth=_SCALE_LINEWIDTH,
        )

    # Draw numeric labels at integer positions
    for rank_value in range(min_rank, max_rank + 1):
        x = _rank_to_position(
            rank_value, textspace, scalewidth, min_rank, max_rank, reverse
        )
        _draw_text(
            ax,
            width,
            height,
            x,
            scale_y - _MAJOR_TICK_HEIGHT / 2 - 0.05,
            str(rank_value),
            ha="center",
            va="bottom",
            size=_SCALE_LABEL_SIZE,
        )


def _compute_line_endpoints(
    ordered_avg_ranks: np.ndarray,
    textspace: float,
    scalewidth: float,
    min_rank: int,
    max_rank: int,
    min_hline_width: float,
    reverse: bool,
) -> tuple[float, float]:
    """
    Compute the fixed x-coordinates where estimator lines end on each side.

    All lines on the same side end at the same x-coordinate to ensure
    vertical alignment of labels. The endpoint is extended if needed to
    accommodate the rank text.

    Parameters
    ----------
    ordered_avg_ranks : np.ndarray
        Average ranks in sorted order.
    textspace : float
        Space reserved for labels.
    scalewidth : float
        Width of the scale.
    min_rank : int
        Lowest rank value on the scale.
    max_rank : int
        Highest rank value on the scale.
    min_hline_width : float
        Minimum horizontal line width to fit rank text.
    reverse : bool
        If True, best ranks go to right side.

    Returns
    -------
    right_side_end : float
        X-coordinate where right-side lines end.
    left_side_end : float
        X-coordinate where left-side lines end.
    """
    n = len(ordered_avg_ranks)
    n_first_half = math.ceil(n / 2)
    first_half = list(range(n_first_half))
    second_half = list(range(n_first_half, n))

    # Determine which estimators go to which side
    right_indices = first_half if reverse else second_half
    left_indices = second_half if reverse else first_half

    # Compute right side endpoint (maximize to fit all lines)
    right_side_end = textspace + scalewidth + 0.2
    for i in right_indices:
        rank_x = _rank_to_position(
            ordered_avg_ranks[i], textspace, scalewidth, min_rank, max_rank, reverse
        )
        right_side_end = max(right_side_end, rank_x + min_hline_width)

    # Compute left side endpoint (minimize to fit all lines)
    left_side_end = textspace - 0.1
    for i in left_indices:
        rank_x = _rank_to_position(
            ordered_avg_ranks[i], textspace, scalewidth, min_rank, max_rank, reverse
        )
        left_side_end = min(left_side_end, rank_x - min_hline_width)

    return right_side_end, left_side_end


def _draw_single_estimator(
    ax: "Axes",
    width: float,
    height: float,
    rank_x: float,
    line_y: float,
    line_end: float,
    goes_right: bool,
    avg_rank: float,
    label: str,
    colour: str,
    scale_y: float,
) -> None:
    """
    Draw the line, rank value, and label for a single estimator.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    width : float
        Figure width.
    height : float
        Figure height.
    rank_x : float
        X-coordinate of the rank position on the scale.
    line_y : float
        Y-coordinate for this estimator's horizontal line.
    line_end : float
        X-coordinate where the horizontal line ends.
    goes_right : bool
        If True, line extends to the right; if False, to the left.
    avg_rank : float
        Average rank value to display.
    label : str
        Estimator name to display.
    colour : str
        Color for the line and text.
    scale_y : float
        Y-coordinate of the scale line.
    """
    # Draw L-shaped line: vertical from scale, then horizontal to endpoint
    _draw_line(
        ax,
        width,
        height,
        [(rank_x, scale_y), (rank_x, line_y), (line_end, line_y)],
        linewidth=_ESTIMATOR_LINEWIDTH,
        color=colour,
    )

    # Draw rank value above the line end
    _draw_text(
        ax,
        width,
        height,
        line_end,
        line_y - 0.075,
        f"{avg_rank:.4f}",
        ha="right" if goes_right else "left",
        va="center",
        size=_RANK_TEXT_SIZE,
        color=colour,
    )

    # Draw label just past the line end
    label_x = line_end + (_LABEL_OFFSET if goes_right else -_LABEL_OFFSET)
    _draw_text(
        ax,
        width,
        height,
        label_x,
        line_y,
        label,
        ha="left" if goes_right else "right",
        va="center",
        size=_LABEL_TEXT_SIZE,
        color=colour,
    )


def _draw_estimators(
    ax: "Axes",
    width: float,
    height: float,
    ordered_avg_ranks: np.ndarray,
    ordered_labels: np.ndarray,
    colours: list[str],
    scale_y: float,
    min_y_offset: float,
    textspace: float,
    scalewidth: float,
    min_rank: int,
    max_rank: int,
    reverse: bool,
) -> None:
    """
    Draw lines and labels for all estimators.

    Estimators are split into two halves: the best-ranked half is displayed
    on one side of the diagram, and the worst-ranked half on the other side.
    The side depends on the `reverse` parameter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    width : float
        Figure width.
    height : float
        Figure height.
    ordered_avg_ranks : np.ndarray
        Average ranks in sorted order (best first).
    ordered_labels : np.ndarray
        Estimator labels in sorted order.
    colours : list of str
        Colors for each estimator.
    scale_y : float
        Y-coordinate of the scale line.
    min_y_offset : float
        Minimum vertical space below scale.
    textspace : float
        Space reserved for labels.
    scalewidth : float
        Width of the scale.
    min_rank : int
        Lowest rank value on the scale.
    max_rank : int
        Highest rank value on the scale.
    reverse : bool
        If True, best ranks appear on the right side.
    """
    # Minimum horizontal line width to fit rank text (6 chars like "1.0000")
    min_hline_width = 6 * (_RANK_TEXT_SIZE / 100) * 0.8

    n = len(ordered_avg_ranks)
    n_first_half = math.ceil(n / 2)

    # Compute fixed line endpoints for vertical label alignment
    right_side_end, left_side_end = _compute_line_endpoints(
        ordered_avg_ranks,
        textspace,
        scalewidth,
        min_rank,
        max_rank,
        min_hline_width,
        reverse,
    )

    # Draw first half (best ranked estimators)
    for i in range(n_first_half):
        line_y = scale_y + min_y_offset + i * _ESTIMATOR_SPACING
        line_end = right_side_end if reverse else left_side_end
        rank_x = _rank_to_position(
            ordered_avg_ranks[i], textspace, scalewidth, min_rank, max_rank, reverse
        )
        _draw_single_estimator(
            ax,
            width,
            height,
            rank_x,
            line_y,
            line_end,
            reverse,
            ordered_avg_ranks[i],
            ordered_labels[i],
            colours[i],
            scale_y,
        )

    # Draw second half (worst ranked estimators)
    for i in range(n_first_half, n):
        line_y = scale_y + min_y_offset + (n - i - 1) * _ESTIMATOR_SPACING
        line_end = left_side_end if reverse else right_side_end
        rank_x = _rank_to_position(
            ordered_avg_ranks[i], textspace, scalewidth, min_rank, max_rank, reverse
        )
        _draw_single_estimator(
            ax,
            width,
            height,
            rank_x,
            line_y,
            line_end,
            not reverse,
            ordered_avg_ranks[i],
            ordered_labels[i],
            colours[i],
            scale_y,
        )


def _draw_cliques(
    ax: "Axes",
    width: float,
    height: float,
    ordered_avg_ranks: np.ndarray,
    cliques: np.ndarray | list,
    scale_y: float,
    textspace: float,
    scalewidth: float,
    min_rank: int,
    max_rank: int,
    reverse: bool,
) -> None:
    """
    Draw horizontal lines connecting estimators in the same clique.

    Clique lines are drawn below the scale, with each clique on a separate
    vertical level to avoid overlap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    width : float
        Figure width.
    height : float
        Figure height.
    ordered_avg_ranks : np.ndarray
        Average ranks in sorted order.
    cliques : np.ndarray or list
        Binary masks indicating clique membership.
    scale_y : float
        Y-coordinate of the scale line.
    textspace : float
        Space reserved for labels.
    scalewidth : float
        Width of the scale.
    min_rank : int
        Lowest rank value on the scale.
    max_rank : int
        Highest rank value on the scale.
    reverse : bool
        If True, adjusts line positioning for reversed scale.
    """
    clique_y = scale_y + _CLIQUE_START_OFFSET
    side_offset = -_CLIQUE_SIDE_OFFSET if reverse else _CLIQUE_SIDE_OFFSET

    for clq in cliques:
        positions = np.where(np.array(clq) == 1)[0]
        if len(positions) == 0:
            continue

        min_idx, max_idx = positions.min(), positions.max()
        x_min = _rank_to_position(
            ordered_avg_ranks[min_idx],
            textspace,
            scalewidth,
            min_rank,
            max_rank,
            reverse,
        )
        x_max = _rank_to_position(
            ordered_avg_ranks[max_idx],
            textspace,
            scalewidth,
            min_rank,
            max_rank,
            reverse,
        )
        _draw_line(
            ax,
            width,
            height,
            [(x_min - side_offset, clique_y), (x_max + side_offset, clique_y)],
            linewidth=_CLIQUE_LINEWIDTH,
        )
        clique_y += _CLIQUE_SPACING


# =============================================================================
# Main Function
# =============================================================================


def plot_critical_difference(
    scores: np.ndarray | list,
    labels: list[str],
    highlight: dict[str, str] | None = None,
    lower_better: bool = False,
    test: str = "wilcoxon",
    correction: str = "holm",
    alpha: float = 0.1,
    width: float = 6,
    textspace: float = 1.5,
    reverse: bool = True,
    return_p_values: bool = False,
) -> tuple["Figure", "Axes"] | tuple["Figure", "Axes", np.ndarray]:
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
    test is sensitive to the estimators included in the test: "For instance the
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

    Parts of the code are adapted from https://github.com/hfawaz/cd-diagram
    with permission from the owner.

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

    # Validate and preprocess inputs
    if isinstance(scores, list):
        scores = np.array(scores)

    if scores.ndim != 2:
        raise ValueError(
            f"scores must be a 2D array of shape (n_datasets, n_estimators), "
            f"got shape {scores.shape}"
        )

    n_datasets, n_estimators = scores.shape

    if len(labels) != n_estimators:
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of estimators "
            f"({n_estimators})"
        )

    if n_estimators < 2:
        raise ValueError("Need at least 2 estimators to compare")

    test = test.lower() if isinstance(test, str) else test
    correction = correction.lower() if isinstance(correction, str) else correction

    if return_p_values and test != "wilcoxon":
        raise ValueError(
            f"Cannot return p values for the {test}, since it does "
            "not calculate p-values."
        )

    # Step 1: Compute ranks and sort estimators
    ranks = _compute_ranks(scores, lower_better)
    ordered_labels, ordered_avg_ranks, ordered_scores = _sort_by_average_rank(
        scores, labels, ranks
    )

    # Step 2: Determine colours for labels
    if highlight is not None:
        colours = [highlight.get(label, "#000000") for label in ordered_labels]
    else:
        colours = ["#000000"] * n_estimators

    # Step 3: Find cliques using statistical tests
    cliques, p_values = _find_cliques(
        ranks,
        ordered_scores,
        ordered_labels,
        ordered_avg_ranks,
        n_datasets,
        n_estimators,
        test,
        correction,
        alpha,
        lower_better,
    )

    # Step 4: Set up figure
    width = float(width)
    textspace = float(textspace)
    height, scale_y, scalewidth, min_y_offset = _compute_figure_dimensions(
        n_estimators, width, textspace
    )
    fig, ax = _create_figure(width, height)

    min_rank = min(1, int(math.floor(min(ordered_avg_ranks))))
    max_rank = max(n_estimators, int(math.ceil(max(ordered_avg_ranks))))

    # Step 5: Draw all components
    _draw_scale(
        ax, width, height, textspace, scalewidth, scale_y, min_rank, max_rank, reverse
    )
    _draw_estimators(
        ax,
        width,
        height,
        ordered_avg_ranks,
        ordered_labels,
        colours,
        scale_y,
        min_y_offset,
        textspace,
        scalewidth,
        min_rank,
        max_rank,
        reverse,
    )
    _draw_cliques(
        ax,
        width,
        height,
        ordered_avg_ranks,
        cliques,
        scale_y,
        textspace,
        scalewidth,
        min_rank,
        max_rank,
        reverse,
    )

    if return_p_values:
        return fig, ax, p_values
    return fig, ax
