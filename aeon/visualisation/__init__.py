"""Plotting utlities for time series."""

__all__ = [
    # Series plotting
    "plot_series",
    "plot_lags",
    "plot_correlations",
    "plot_windows",
    "plot_time_series_with_change_points",
    "plot_time_series_with_profiles",
    # Results plotting
    "plot_critical_difference",
    "plot_boxplot_median",
    "plot_scatter_predictions",
    "plot_scatter",
    # Estimator plotting
    "plot_cluster_algorithm",
    "plot_temporal_importance_curves",
]

from aeon.visualisation.estimator._cluster_plotting import plot_cluster_algorithm
from aeon.visualisation.estimator._temporal_importance_curves import (
    plot_temporal_importance_curves,
)
from aeon.visualisation.results._boxplot import plot_boxplot_median
from aeon.visualisation.results._critical_difference import plot_critical_difference
from aeon.visualisation.results._scatter import plot_scatter, plot_scatter_predictions
from aeon.visualisation.series._segmentation import (
    plot_time_series_with_change_points,
    plot_time_series_with_profiles,
)
from aeon.visualisation.series._series import (
    plot_correlations,
    plot_interval,
    plot_lags,
    plot_series,
    plot_windows,
)
