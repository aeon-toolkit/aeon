"""Testing index functions."""

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from aeon.testing.utils.data_gen import _make_hierarchical
from aeon.utils.index_functions import (
    _get_cutoff_from_index,
    get_cutoff,
    get_slice,
    get_time_index,
    get_window,
)

cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(2)]

Xlist = [
    pd.DataFrame([[0, 0, 1, 4], [0, 1, 2, 5], [0, 2, 3, 6]], columns=cols),
    pd.DataFrame([[1, 0, 1, 4], [1, 1, 2, 55], [1, 2, 3, 6]], columns=cols),
    pd.DataFrame([[2, 0, 1, 42], [2, 1, 2, 5], [2, 2, 3, 6]], columns=cols),
]
X = pd.concat(Xlist)
multi_index = X.set_index(["instances", "timepoints"])

cols = ["foo", "bar", "timepoints"] + [f"var_{i}" for i in range(2)]
Xlist = [
    pd.DataFrame(
        [["a", 0, 0, 1, 4], ["a", 0, 1, 2, 5], ["a", 0, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["a", 1, 0, 1, 4], ["a", 1, 1, 2, 55], ["a", 1, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["a", 2, 0, 1, 42], ["a", 2, 1, 2, 5], ["a", 2, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 0, 0, 1, 4], ["b", 0, 1, 2, 5], ["b", 0, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 1, 0, 1, 4], ["b", 1, 1, 2, 55], ["b", 1, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 2, 0, 1, 42], ["b", 2, 1, 2, 5], ["b", 2, 2, 3, 6]], columns=cols
    ),
]
X = pd.concat(Xlist)
multi_index_hier = X.set_index(["foo", "bar", "timepoints"])

EXAMPLE_DATA = {
    "pd.Series": pd.Series(np.random.rand(4)),
    "pd.DataFrame": pd.DataFrame(np.random.rand(4, 2)),
    "np.ndarray": np.random.rand(4, 2),
    "numpy3D": np.random.rand(4, 2, 3),
    "pd-multiindex": multi_index,
    "pd-multiindex-hier": multi_index_hier,
}


@pytest.mark.parametrize("datatype", EXAMPLE_DATA.keys())
def test_get_time_index(datatype):
    """Tests that conversions agree with input data."""
    data = EXAMPLE_DATA[datatype]
    idx = get_time_index(data)

    msg = f"get_time_index should return pd.Index, but found {type(idx)}"
    assert isinstance(idx, pd.Index), msg

    if datatype in ["pd.Series", "pd.DataFrame"]:
        assert (idx == data.index).all()

    if datatype in ["np.ndarray", "numpy3D"]:
        assert isinstance(idx, pd.RangeIndex)
        if datatype == "np.ndarray":
            assert len(idx) == data.shape[0]
        else:
            assert len(idx) == data.shape[-1]
    if isinstance(data, pd.MultiIndex):
        assert isinstance(idx, pd.Index)
        assert (idx == data.get_level_values(-1)).all()
    if datatype in ["pd-multiindex", "pd_multiindex_hier"]:
        exp_idx = data.index.get_level_values(-1).unique()
        assert (idx == exp_idx).all()


@pytest.mark.parametrize("convert_input", [True, False])
@pytest.mark.parametrize("reverse_order", [True, False])
@pytest.mark.parametrize("return_index", [True, False])
@pytest.mark.parametrize("datatype", EXAMPLE_DATA.keys())
def test_get_cutoff(datatype, return_index, reverse_order, convert_input):
    """Tests that get_cutoff has correct output.

    Parameters
    ----------
    datatype : str - datatype of input
    return_index : bool - whether index (True) or index element is returned (False)
    reverse_order : bool - whether first (True) or last index (False) is retrieved
    convert_input : bool - whether input is converted (True) or passed through (False)

    Raises
    ------
    AssertionError if get_cutoff does not return a length 1 pandas.index
        for any fixture example of given scitype, mtype
    """
    # retrieve example data structure
    data = EXAMPLE_DATA[datatype]

    cutoff = get_cutoff(
        data,
        return_index=return_index,
        reverse_order=reverse_order,
        convert_input=convert_input,
    )

    if return_index:
        expected_types = pd.Index
        cutoff_val = cutoff[0]
    else:
        expected_types = (int, float, np.int64, pd.Timestamp)
        cutoff_val = cutoff

    msg = (
        f"incorrect return type of get_cutoff"
        f"expected {expected_types}, found {type(cutoff)}"
    )

    assert isinstance(cutoff, expected_types), msg

    if return_index:
        assert len(cutoff) == 1
        if isinstance(cutoff_val, (pd.Period, pd.Timestamp)):
            assert hasattr(cutoff, "freq") and cutoff.freq is not None

    if isinstance(data, np.ndarray):
        if reverse_order:
            assert cutoff_val == 0
        else:
            assert cutoff_val > 0

    if datatype in ["pd.Series", "pd.DataFrame"]:
        if reverse_order:
            assert cutoff_val == data.index[0]
        else:
            assert cutoff_val == data.index[-1]

    if datatype in ["pd-multiindex", "pd_multiindex_hier"]:
        time_idx = data.index.get_level_values(-1)
        if reverse_order:
            assert cutoff_val == time_idx.min()
        else:
            assert cutoff_val == time_idx.max()


@pytest.mark.parametrize("reverse_order", [True, False])
def test_get_cutoff_from_index(reverse_order):
    """Tests that _get_cutoff_from_index has correct output.

    Parameters
    ----------
    return_index : bool - whether index (True) or index element is returned (False)
    reverse_order : bool - whether first (True) or last index (False) is retrieved

    Raises
    ------
    AssertionError if _get_cutoff_from_index does not return a length 1 pandas.index
    AssertionError if _get_cutoff_from_index does not return the correct cutoff value
    """
    hier_fixture = _make_hierarchical()
    hier_idx = hier_fixture.index

    cutoff = _get_cutoff_from_index(
        hier_idx, return_index=True, reverse_order=reverse_order
    )
    idx = _get_cutoff_from_index(
        hier_idx, return_index=False, reverse_order=reverse_order
    )

    assert isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) == 1
    assert cutoff.freq == "D"
    assert idx == cutoff[0]

    if reverse_order:
        assert idx == pd.Timestamp("2000-01-01")
    else:
        assert idx == pd.Timestamp("2000-01-12")
    series_fixture = EXAMPLE_DATA["pd.Series"]
    series_idx = series_fixture.index

    cutoff = _get_cutoff_from_index(
        series_idx, return_index=True, reverse_order=reverse_order
    )
    idx = _get_cutoff_from_index(
        series_idx, return_index=False, reverse_order=reverse_order
    )

    assert isinstance(cutoff, pd.Index) and len(cutoff) == 1
    assert is_integer_dtype(cutoff)
    assert idx == cutoff[0]

    if reverse_order:
        assert idx == 0
    else:
        assert idx == 3


@pytest.mark.parametrize("bad_inputs", ["foo", 12345, [[[]]]])
def test_get_cutoff_wrong_input(bad_inputs):
    """Tests that get_cutoff raises error on bad input when input checks are enabled.

    Parameters
    ----------
    bad_inputs : inputs that should set off the input checks

    Raises
    ------
    Exception (from pytest) if the error is not raised as expected
    """
    with pytest.raises(Exception, match="must be of Series, Panel, or Hierarchical"):
        get_cutoff(bad_inputs, check_input=True)


@pytest.mark.parametrize("window_length, lag", [(2, 0), (None, 0), (4, 1)])
@pytest.mark.parametrize("datatype", EXAMPLE_DATA.keys())
def test_get_window_output_type(datatype, window_length, lag):
    """Tests that get_window runs for all mtypes, and returns output of same mtype.

    Parameters
    ----------
    datatype : str - datatype of input
    window_length : int, passed to get_window
    lag : int, passed to get_window

    Raises
    ------
    Exception if get_window raises one
    """
    # retrieve example fixture
    data = EXAMPLE_DATA[datatype]
    X = get_window(data, window_length=window_length, lag=lag)
    assert isinstance(X, type(data))


def test_get_window_expected_result():
    """Tests that get_window produces return of the right length."""
    X_df = EXAMPLE_DATA["pd.DataFrame"]
    assert len(get_window(X_df, 2, 1)) == 2
    assert len(get_window(X_df, 3, 1)) == 3
    assert len(get_window(X_df, 1, 2)) == 1
    assert len(get_window(X_df, 3, 4)) == 0
    assert len(get_window(X_df, 3, None)) == 3
    assert len(get_window(X_df, None, 2)) == 2
    assert len(get_window(X_df, None, None)) == 4
    X_mi = EXAMPLE_DATA["pd-multiindex"]
    assert len(get_window(X_mi, 3, 1)) == 6
    assert len(get_window(X_mi, 2, 0)) == 6
    assert len(get_window(X_mi, 2, 4)) == 0
    assert len(get_window(X_mi, 1, 2)) == 3
    assert len(get_window(X_mi, 2, None)) == 6
    assert len(get_window(X_mi, None, 2)) == 3
    assert len(get_window(X_mi, None, None)) == 9
    X_hi = EXAMPLE_DATA["pd-multiindex-hier"]
    assert len(get_window(X_hi, 3, 1)) == 12
    assert len(get_window(X_hi, 2, 0)) == 12
    assert len(get_window(X_hi, 2, 4)) == 0
    assert len(get_window(X_hi, 1, 2)) == 6
    assert len(get_window(X_hi, 2, None)) == 12
    assert len(get_window(X_hi, None, 2)) == 6
    assert len(get_window(X_hi, None, None)) == 18
    X_np3d = np.random.rand(3, 2, 3)
    assert get_window(X_np3d, 3, 1).shape == (2, 2, 3)
    assert get_window(X_np3d, 2, 0).shape == (2, 2, 3)
    assert get_window(X_np3d, 2, 4).shape == (0, 2, 3)
    assert get_window(X_np3d, 1, 2).shape == (1, 2, 3)
    assert get_window(X_np3d, 2, None).shape == (2, 2, 3)
    assert get_window(X_np3d, None, 2).shape == (1, 2, 3)
    assert get_window(X_np3d, None, None).shape == (3, 2, 3)


@pytest.mark.parametrize("datatype", EXAMPLE_DATA.keys())
def test_get_slice_output_type(datatype):
    """Tests that get_slice runs for all mtypes, and returns output of same mtype.

    Parameters
    ----------
    scitype : str - scitype of input
    mtype : str - mtype of input

    Raises
    ------
    Exception if get_slice raises one
    """
    # retrieve example fixture
    data = EXAMPLE_DATA[datatype]
    X = get_slice(data)
    assert isinstance(X, type(data))


def test_get_slice_expected_result():
    """Tests that get_slice produces return of the right length.

    Raises
    ------
    Exception if get_slice raises one
    """
    X_df = EXAMPLE_DATA["pd.DataFrame"]
    assert len(get_slice(X_df, start=1, end=3)) == 2

    X_s = EXAMPLE_DATA["pd.Series"]
    assert len(get_slice(X_s, start=1, end=3)) == 2

    X_np = EXAMPLE_DATA["numpy3D"]
    assert get_slice(X_np, start=1, end=3).shape == (2, 2, 3)
