"""Utility class for ploting functionality."""

__all__ = [
    "plot_series_with_change_points",
]

__maintainer__ = []

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.utils.validation.series import check_series


def plot_series_with_change_points(y, change_points, title=None, font_size=16):
    """Plot the time series with the known change points.

    Parameters
    ----------
    y: array-like, shape = [n]
        the univariate time series of length n to be annotated
    change_points: array-like, dtype=int
        the known change points
        these are highlighted in the time series as vertical lines
    title: str, default=None
        the name of the time series (dataset) to be annotated
    font_size: int, default=16
        for plotting

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : plt.Axis

    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    y = check_series(y)

    true_cps = np.sort(change_points)
    segments = [0] + list(true_cps) + [y.shape[0] - 1]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.25))
    for idx in np.arange(0, len(segments) - 1):
        ax.plot(
            range(segments[idx], segments[idx + 1] + 1),
            y[segments[idx] : segments[idx + 1] + 1],
        )

    lim1 = plt.ylim()[0]
    lim2 = plt.ylim()[1]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(true_cps)))
    for i, idx in enumerate(true_cps):
        ax.vlines(idx, lim1, lim2, linestyles="--", label=f"ChP {i}", color=colors[i])

    plt.legend(loc="best")

    # Set the figure's title
    if title is not None:
        fig.suptitle(title, fontsize=font_size)

    return fig, ax
