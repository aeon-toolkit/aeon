"""Legacy data generators."""

__all__ = [
    "make_series",
    "make_forecasting_problem",
    "_make_index",
    "get_examples",
]

from aeon.testing.data_generation._legacy._series import (
    _make_index,
    make_forecasting_problem,
    make_series,
)
from aeon.testing.data_generation._legacy._test_examples import get_examples
