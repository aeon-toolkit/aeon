# -*- coding: utf-8 -*-
"""Plotting tools for estimator results."""
__all__ = ["plot_critical_difference"]

from aeon.benchmarking._critical_difference import plot_critical_difference


def plot_boxplot_median(results, names, plot_type="violin"):
    """Plot a box plot of distributions from the median.

    Each row of results is an independent experiment for each element in names. This
    function works out the deviation from the median for each row, then plots a
    boxplot variant of each column.
    """
    pass
