"""Plotting utlities for time series."""

__all__ = [
    # Series plotting
    "plot_series",
    "plot_interval",
    "plot_lags",
    "plot_correlations",
    "plot_windows",
    # Results plotting
    "plot_critical_difference",
    "plot_boxplot_median",
    "plot_scatter_predictions",
    "plot_scatter",
    # Segmentation plotting
    "plot_time_series_with_change_points",
    "plot_time_series_with_profiles",
    # Clustering plotting
    "plot_cluster_algorithm",
]
from aeon.visualisation._cluster_plotting import plot_cluster_algorithm
from aeon.visualisation._critical_difference import plot_critical_difference
from aeon.visualisation.results_plotting import (
    plot_boxplot_median,
    plot_scatter,
    plot_scatter_predictions,
)
from aeon.visualisation.segmentation_plotting import (
    plot_time_series_with_change_points,
    plot_time_series_with_profiles,
)
from aeon.visualisation.series_plotting import (
    plot_correlations,
    plot_interval,
    plot_lags,
    plot_series,
    plot_windows,
)
