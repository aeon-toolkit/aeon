# -*- coding: utf-8 -*-
"""Benchmarking."""
__all__ = [
    "plot_critical_difference",
    "get_available_estimators",
    "get_estimator_results",
    "get_estimator_results_as_array",
]

from aeon.benchmarking._critical_difference import plot_critical_difference
from aeon.benchmarking.results_loaders import (
    get_available_estimators,
    get_estimator_results,
    get_estimator_results_as_array,
)
