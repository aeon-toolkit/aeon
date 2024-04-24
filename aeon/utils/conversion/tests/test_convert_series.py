"""Tests for convert series."""

import numpy as np
import pandas as pd
import pytest

from aeon.utils.conversion import convert_series
from aeon.utils.conversion._convert_series import (
    _resolve_input_type,
    _resolve_output_type,
)

TO_TYPE = ["np.ndarray", "pd.Series", "pd.DataFrame"]
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
    """Test convert a univariate series stored in 2D data structure."""
    x = convert_series(data, to_type)
    assert isinstance(x, eval(to_type))


@pytest.mark.parametrize("data", MULTIVARIATE)
@pytest.mark.parametrize("to_type", TO_TYPE)
def test_convert_multivariate_series(data, to_type):
    """Test convert series with single input of each type."""
    if to_type == "pd.Series":
        with pytest.raises(ValueError, match="cannot convert to pd.Series"):
            convert_series(data, to_type)
        return
    x = convert_series(data, to_type)
    assert isinstance(x, eval(to_type))


def test_convert_series_wrong_type():
    """Test convert series with wrong input type."""
    with pytest.raises(ValueError, match="Unknown input type"):
        convert_series("string", "np.ndarray")
    with pytest.raises(ValueError, match="Unknown input type"):
        convert_series([1, 2, 3], "np.ndarray")
    with pytest.raises(ValueError, match="Unknown output type"):
        convert_series(np.array([[[1, 2, 3]]]), "numpy3D")
    with pytest.raises(ValueError, match="Unknown output type"):
        convert_series(np.ndarray([1, 2, 3]), "np-list")
    with pytest.raises(ValueError, match="Unknown output type"):
        convert_series(np.ndarray([1, 2, 3]), "pd_multiindex_hier")


def test__resolve_input_type():
    """Test _input_type function."""
    for i in range(len(UNIVARIATE)):
        assert _resolve_input_type(UNIVARIATE[i]) == TO_TYPE[i]
    with pytest.raises(ValueError, match="Unknown input type"):
        _resolve_input_type("string")


def test__resolve_output_type():
    """Test resolve output type."""
    for i in range(len(TO_TYPE)):
        assert _resolve_output_type(TO_TYPE[i]) == TO_TYPE[i]
    with pytest.raises(ValueError, match="Unknown output type"):
        _resolve_output_type("string")
    test = ["np.ndarray", "pd.Series", "pd.DataFrame"]
    assert _resolve_output_type(test) == "np.ndarray"
    test = ["FOO", "pd.Series", "pd.DataFrame"]
    assert _resolve_output_type(test) == "pd.Series"
    test = ["FOO", "BAR", "pd.DataFrame"]
    assert _resolve_output_type(test) == "pd.DataFrame"
    test = ["FOO", "BAR", "ARSENAL"]
    with pytest.raises(ValueError, match="Unknown input type"):
        _resolve_input_type(test)


def test_convert_series_lists():
    """Test series conversion with list type inputs."""
    # Test returns self if in list
    for i in range(len(UNIVARIATE)):
        test = ["np.ndarray", "pd.Series", "pd.DataFrame"]
        x = convert_series(UNIVARIATE[i], test)
        assert type(x) is type(UNIVARIATE[i])
        # Test with invalid names at start
        test.insert(0, "FOO")
        x = convert_series(UNIVARIATE[i], test)
        assert type(x) is type(UNIVARIATE[i])
        # Test when type not present
        test = ["np.ndarray", "pd.Series", "pd.DataFrame"]
        test.remove(TO_TYPE[i])
        x = convert_series(UNIVARIATE[i], test)
        assert x.__class__.__name__ == test[0].split(".")[1]
        # Test with invalid names at start
        test.insert(0, "FOO")
        x = convert_series(UNIVARIATE[i], test)
        assert x.__class__.__name__ == test[1].split(".")[1]


def test_convert_series_single_element():
    """Test a DataFrame with a single element is correctly converted to a pd.Series."""
    x = pd.DataFrame([1])
    x = pd.DataFrame({"0": [10]}, index=pd.to_datetime(["2000-02-18"]))
    y = convert_series(x, "pd.Series")
    assert isinstance(y, pd.Series)
    assert isinstance(y.index, pd.DatetimeIndex)
