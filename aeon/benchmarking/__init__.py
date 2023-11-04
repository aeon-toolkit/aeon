"""Benchmarking."""
__all__ = [
    "plot_critical_difference",
    "get_available_estimators",
    "get_estimator_results",
    "get_estimator_results_as_array",
    "get_bake_off_2017_results",
    "classifiers_2017",
]

from aeon.benchmarking._critical_difference import plot_critical_difference
from aeon.benchmarking.results_loaders import (
    classifiers_2017,
    get_available_estimators,
    get_bake_off_2017_results,
    get_estimator_results,
    get_estimator_results_as_array,
)
