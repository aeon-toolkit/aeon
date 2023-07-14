# -*- coding: utf-8 -*-
"""Time series averaging metrics."""
__all__ = ["elastic_barycenter_average", "mean_average", "_resolve_average_callable"]
from aeon.clustering.metrics.averaging._averaging import (
    _resolve_average_callable,
    elastic_barycenter_average,
    mean_average,
)
