"""Plotting utlities for time series."""

__all__ = [
    # Series plotting
    "plot_series",
    "plot_lags",
    "plot_correlations",
    "plot_series_collection",
    "plot_collection_by_class",
    "plot_spectrogram",
    # Learning task plotting
    "plot_series_windows",
    "plot_series_with_change_points",
    # Results plotting
    "plot_critical_difference",
    "plot_significance",
    "plot_boxplot_median",
    "plot_scatter_predictions",
    "plot_pairwise_scatter",
    "plot_score_vs_time_scatter",
    # Estimator plotting
    "plot_series_with_profiles",
    "plot_cluster_algorithm",
    "plot_temporal_importance_curves",
]

from aeon.visualisation.estimator._clasp import plot_series_with_profiles
from aeon.visualisation.estimator._clustering import plot_cluster_algorithm
from aeon.visualisation.estimator._temporal_importance_curves import (
    plot_temporal_importance_curves,
)
from aeon.visualisation.learning_task._forecasting import plot_series_windows
from aeon.visualisation.learning_task._segmentation import (
    plot_series_with_change_points,
)
from aeon.visualisation.results._boxplot import plot_boxplot_median
from aeon.visualisation.results._critical_difference import plot_critical_difference
from aeon.visualisation.results._scatter import (
    plot_pairwise_scatter,
    plot_scatter_predictions,
    plot_score_vs_time_scatter,
)
from aeon.visualisation.results._significance import plot_significance
from aeon.visualisation.series._collections import (
    plot_collection_by_class,
    plot_series_collection,
)
from aeon.visualisation.series._series import (
    plot_correlations,
    plot_lags,
    plot_series,
    plot_spectrogram,
)
