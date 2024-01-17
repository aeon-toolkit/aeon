"""Temporal importance curve diagram generators for interval forests."""

__author__ = ["MatthewMiddlehurst"]

__all__ = ["plot_temporal_importance_curves"]

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_temporal_importance_curves(
    curves, curve_names, top_curves_shown=None, plot_mean=True
):
    """Temporal importance curve diagram generator for interval forests."""
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

    # plot curves with highest max and the mean information gain for each time point if
    # enabled.
    for i in range(0, top_curves_shown):
        plt.plot(
            top_curves[i],
            label=top_names[i],
        )
    if plot_mean:
        plt.plot(
            list(np.mean(curves, axis=0)),
            "--",
            linewidth=3,
            label="Mean Information Gain",
        )
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.xlabel("Time Point")
    plt.ylabel("Information Gain")

    plt.show()
