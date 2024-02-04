__all__ = [
    "plot_series_collection",
    "plot_collection_by_class",
]

import numpy as np

from aeon.utils.conversion import convert_collection
from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_series_collection(X, y=None):
    """Plot a collection of time series.

    Plot each time series in a collection. Can accept any aeon collection format.

    Parameters
    ----------
    X : array-like of shape (n_samples, 1, n_features)
        The collection of time series to plot.
    y : array-like of shape (n_samples,), default=None
        The class labels for each time series. If present, each series will be
        colored according to its class label.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    X = convert_collection(X, "np-list")
    has_class_labels = y is not None

    unique_labels = sorted(set(y)) if has_class_labels else {None}
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    for i, series in enumerate(X):
        label = y[i] if has_class_labels else None
        color = label_color_map[label] if has_class_labels else None
        ax.plot(series[0], label=label, color=color)

    if has_class_labels:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_map = dict(zip(labels, handles))
        ax.legend(
            [label_handle_map[str(label)] for label in unique_labels], unique_labels
        )

    return fig, ax


def plot_collection_by_class(X, y):
    """Plot a collection of time series, grouped by class.

    Each class is plotted in a separate subplot with a single legend entry for the
    class label. Can accept any aeon collection format.

    Parameters
    ----------
    X : array-like of shape (n_samples, 1, n_features)
        The collection of time series to plot.
    y : array-like of shape (n_samples,)
        The class labels for each time series.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    X = convert_collection(X, "np-list")

    unique_labels = sorted(set(y))
    n_classes = len(unique_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    fig, axs = plt.subplots(n_classes, 1, figsize=(12, 3 * n_classes), sharex=True)
    if n_classes == 1:  # If there's only one class, axs is not a list
        axs = [axs]

    for label, ax, color in zip(unique_labels, axs, colors):
        plotted = False
        for series, class_label in zip(X, y):
            if class_label == label:
                if not plotted:
                    ax.plot(series[0], color=color, label=f"Class {label}")
                    plotted = True
                else:
                    ax.plot(series[0], color=color)
        ax.legend()

    return fig, axs
