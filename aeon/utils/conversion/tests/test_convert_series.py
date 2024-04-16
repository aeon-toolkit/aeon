"""Tests for convert series."""

import numpy as np
import pandas as pd
import pytest

from aeon.utils.conversion import convert_series

TO_TYPE = ["pd.DataFrame", "pd.Series", "np.ndarray"]
uni = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
uni2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
UNIVARIATE = [uni, pd.Series(uni), pd.DataFrame(uni)]
UNIVARIATE2 = [uni2, pd.DataFrame(uni2), uni2.T, pd.DataFrame(uni2.T)]
multi = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
MULTIVARIATE = [multi, pd.DataFrame(multi)]


@pytest.mark.parametrize("data", UNIVARIATE)
@pytest.mark.parametrize("to_type", TO_TYPE)
def test_convert_univariate_series(data, to_type):
    """Test convert series with single univariate input of each type."""
    x = convert_series(data, to_type)
    assert isinstance(x, eval(to_type))


@pytest.mark.parametrize("data", UNIVARIATE2)
@pytest.mark.parametrize("to_type", TO_TYPE)
def test_convert_univariate_as_multivariate(data, to_type):
    """Test univariate convert series stored in 2D data structure."""
    x = convert_series(data, to_type)
    assert isinstance(x, eval(to_type))


@pytest.mark.parametrize("data", MULTIVARIATE)
@pytest.mark.parametrize("to_type", TO_TYPE)
def test_convert_multivariate_series(data, to_type):
    """Test convert series with single input of each type."""
    x = convert_series(data, to_type)
    assert isinstance(x, eval(to_type))
