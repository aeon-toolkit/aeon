"""Cluster plotting tools."""

__maintainer__ = []

__all__ = ["plot_cluster_algorithm"]

import numpy as np

from aeon.clustering.base import BaseClusterer
from aeon.utils.conversion import convert_collection
from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_cluster_algorithm(model: BaseClusterer, X, k: int):
    """Plot the results from a univariate partitioning algorithm.

    Parameters
    ----------
    model: BaseClusterer
        Clustering model to plot
    X: np.ndarray or pd.Dataframe or List[pd.Dataframe]
        The series to predict the values for
    k: int
        Number of centers

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    predict_series = convert_collection(X, "numpy3D")

    plt.figure(figsize=(5, 10))
    plt.rcParams["figure.dpi"] = 100
    indexes = model.predict(predict_series)

    centers = model.cluster_centers_
    series_values = _get_cluster_values(indexes, predict_series, k)

    fig, axes = plt.subplots(nrows=k, ncols=1)
    for i in range(k):
        _plot(series_values[i], centers[i], axes[i])

    blue_patch = mpatches.Patch(color="blue", label="Series that belong to the cluster")
    red_patch = mpatches.Patch(color="red", label="Cluster centers")
    plt.legend(
        handles=[red_patch, blue_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.tight_layout()

    return fig, axes


def _plot(cluster_values, center, axes):
    for cluster_series in cluster_values:
        for cluster in cluster_series:
            axes.plot(cluster, color="b")
    axes.plot(center[0], color="r")


def _get_cluster_values(cluster_indexes: np.ndarray, X: np.ndarray, k: int):
    ts_in_center = []
    for i in range(k):
        curr_indexes = np.where(cluster_indexes == i)[0]
        ts_in_center.append(X[curr_indexes])

    return ts_in_center
