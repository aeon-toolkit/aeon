"""Temporal importance curve diagram generators for interval forests."""

__maintainer__ = []

__all__ = ["plot_temporal_importance_curves"]

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_temporal_importance_curves(
    curves, curve_names, top_curves_shown=None, plot_mean=True
):
    """Temporal importance curve diagram generator for interval forests.

    Parameters
    ----------
    curves : larray-like of shape (n_curves, n_timepoints)
        The temporal importance curves for each attribute.
    curve_names : list of str of shape (n_curves)
        The names of the attributes.
    top_curves_shown : int, default=None
        The number of curves to show. If None, all curves are shown.
    plot_mean : bool, default=True
        Whether to plot the mean temporal importance curve.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis
    """
    # find attributes to display by max information gain for any time point.
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    top_curves_shown = len(curves) if top_curves_shown is None else top_curves_shown
    max_ig = [max(i) for i in curves]
    top = sorted(range(len(max_ig)), key=lambda i: max_ig[i], reverse=True)[
        :top_curves_shown
    ]

    top_curves = [curves[i] for i in top]
    top_names = [curve_names[i] for i in top]

    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.7))

    # plot curves with highest max and the mean information gain for each time point if
    # enabled.
    for i in range(0, top_curves_shown):
        ax.plot(
            top_curves[i],
            label=top_names[i],
        )
    if plot_mean:
        ax.plot(
            list(np.mean(curves, axis=0)),
            "--",
            linewidth=3,
            label="Mean Information Gain",
        )
    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Information Gain")

    return fig, ax
