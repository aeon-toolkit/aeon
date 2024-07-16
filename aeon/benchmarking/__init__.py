"""Benchmarking."""

__all__ = [
    "get_available_estimators",
    "get_estimator_results",
    "get_estimator_results_as_array",
    "get_bake_off_2017_results",
    "get_bake_off_2021_results",
    "get_bake_off_2023_results",
    "uni_classifiers_2017",
    "multi_classifiers_2021",
    "uni_classifiers_2023",
    "get_available_estimators",
]

from aeon.benchmarking.results_loaders import (
    get_available_estimators,
    get_bake_off_2017_results,
    get_bake_off_2021_results,
    get_bake_off_2023_results,
    get_estimator_results,
    get_estimator_results_as_array,
    multi_classifiers_2021,
    uni_classifiers_2017,
    uni_classifiers_2023,
)
