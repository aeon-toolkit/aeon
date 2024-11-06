"""Test series module."""

__maintainer__ = ["TonyBagnall"]

import numpy as np
import pandas as pd

from aeon.testing.data_generation import _make_hierarchical
from aeon.utils.validation.series import is_hierarchical, is_univariate_series


def test_is_univariate_series():
    """Test Univariate Series."""
    assert not is_univariate_series(None)
    assert is_univariate_series(pd.Series([1, 2, 3, 4, 5]))
    assert is_univariate_series(np.array([1, 2, 3, 4, 5]))
    assert is_univariate_series(pd.DataFrame({"A": [1, 2, 3, 4, 5]}))
    assert not is_univariate_series(
        pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    )
    assert not is_univariate_series(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))


def test_is_hierarchical():
    """Test MultiIndex Series."""
    series = _make_hierarchical()
    assert is_hierarchical(series)
    assert not is_hierarchical(
        pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    )
    assert not is_hierarchical(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
