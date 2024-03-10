"""Tests for base series estimator."""

import numpy as np
import pandas as pd
import pytest

from aeon.base import BaseSeriesEstimator


def test__check_X():
    """Test if capabilities correctly tested."""
    dummy1 = BaseSeriesEstimator()
    invalid_always = np.array(["1", "2", "3", "4", "5"])
    with pytest.raises(ValueError, match=r"Error in input type should be one of"):
        dummy1._check_X("String input")
    with pytest.raises(ValueError, match=r"array must contain floats or ints"):
        dummy1._check_X(invalid_always)
    with pytest.raises(ValueError, match=r"pd.Series must be numeric"):
        dummy1._check_X(pd.Series(invalid_always))
    with pytest.raises(ValueError, match=r"pd.DataFrame must be numeric"):
        dummy1._check_X(pd.DataFrame(invalid_always))
    multi_np = np.random.random(size=(5, 10))
    multi_pd = pd.DataFrame(multi_np)
    with pytest.raises(ValueError, match=r"Multivariate data not supported"):
        dummy1._check_X(multi_np)
    with pytest.raises(ValueError, match=r"Multivariate data not supported"):
        dummy1._check_X(multi_pd)
    dummy2 = BaseSeriesEstimator()
    all_tags = {
        "capability:multivariate": True,
        "capability:missing_values": True,
    }
    dummy2.set_tags(**all_tags)
    meta = dummy2._check_X(multi_np)
    meta2 = dummy2._check_X(multi_pd)
    assert meta == meta2
    assert meta["multivariate"]
    assert not meta["missing_values"]
    multi_np[0][0] = np.NAN
    multi_pd[0][0] = np.NAN
    uni_missing = pd.Series(np.array([1.0, np.NAN, 2.0]))
    meta = dummy2._check_X(multi_np)
    meta2 = dummy2._check_X(multi_pd)
    meta3 = dummy2._check_X(uni_missing)
    assert (
        meta["missing_values"] and meta2["missing_values"] and meta3["missing_values"]
    )
    work_always = np.random.random(size=(10))
    meta = dummy1._check_X(work_always)
    meta2 = dummy1._check_X(pd.Series(work_always))
    meta3 = dummy1._check_X(pd.DataFrame(work_always))
    assert meta == meta2 == meta3
    all_tags = {"capability:univariate": False}
    dummy2.set_tags(**all_tags)
    uni_np = np.random.random(size=(10))
    uni_pd = pd.Series(uni_np)
    with pytest.raises(ValueError, match=r"Univariate data not supported"):
        dummy2._check_X(uni_np)
    with pytest.raises(ValueError, match=r"Univariate data not supported"):
        dummy2._check_X(uni_pd)


UNIVARIATE = {
    "np.ndarray": np.random.random(size=(20)),
    "pd.Series": pd.Series(np.random.random(size=(20))),
    "pd.DataFrame": pd.DataFrame(np.random.random(size=(20))),
}
MULTIVARIATE = {
    "np.ndarray": np.random.random(size=(5, 20)),
    "pd.Series": pd.Series(np.random.random(size=(20))),
    "pd.DataFrame": pd.DataFrame(np.random.random(size=(5, 20))),
}
VALID_TYPES = [
    "np.ndarray",
    "pd.Series",
    "pd.DataFrame",
]


def test_univariate_convert_X():
    """Test _convert_X on univariate series."""
    # np.ndarray inner, no multivariate. Univariate are always series,
    # not dataframes, axis is ignored
    dummy1 = BaseSeriesEstimator()
    for st in ["np.ndarray", "pd.Series"]:  # Inner type, DataFrame not allowed for
        # univariate
        for st2 in VALID_TYPES:  # input type
            X = UNIVARIATE[st2]
            dummy1.set_tags(**{"X_inner_type": st})
            X2 = dummy1._convert_X(X, axis=0)
            assert type(X2).__name__ == st.split(".")[1]
            X2 = dummy1._convert_X(X, axis=1)
            assert len(X) == len(X2)


def test_multivariate_convert_X():
    """Test _convert_X on multivariate series."""
    # np.ndarray inner multivariate capable. Multivariate are always DataFrames,
    # not Series. 1D np.ndarray converted to 2D. Capability is tested in check_X,
    # not here
    dummy1 = BaseSeriesEstimator(axis=0)
    for st in ["np.ndarray", "pd.DataFrame"]:  # Inner type, Series not allowed for
        # Multivariate
        name = st.split(".")[1]
        for st2 in VALID_TYPES:  # input type
            dummy1.set_tags(**{"X_inner_type": st, "capability:multivariate": True})
            X = MULTIVARIATE[st2]
            X2 = dummy1._convert_X(X, axis=0)
            assert type(X2).__name__ == name

            shape1 = X2.shape
            X2 = dummy1._convert_X(X, axis=1)
            shape2 = X2.shape
            assert shape1[0] == shape2[1] and shape1[1] == shape2[0]


@pytest.mark.parametrize("input_type", VALID_TYPES)
@pytest.mark.parametrize("inner_type", VALID_TYPES)
def test_preprocess_series(input_type, inner_type):
    """Test _preprocess_series."""
    dummy1 = BaseSeriesEstimator(axis=0)
    inner = inner_type.split(".")[1]
    if inner_type != "pd.DataFrame":
        dummy1.set_tags(**{"X_inner_type": inner_type})
        X = UNIVARIATE[input_type]
        X2 = dummy1._preprocess_series(X)
        assert type(X2).__name__ == inner
    if inner_type != "pd.Series":
        dummy1.set_tags(**{"X_inner_type": inner_type, "capability:multivariate": True})
        X = UNIVARIATE[input_type]
        X2 = dummy1._preprocess_series(X)
        assert type(X2).__name__ == inner


INPUT_CORRECT = [
    np.array([0, 0, 0, 1, 1]),
    pd.Series([0, 0, 0, 1, 1]),
    pd.DataFrame([0, 0, 0, 1, 1]),
]
INPUT_WRONG = [
    np.array([[0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2]]),
    pd.DataFrame([[0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2]]),
    np.array([0, 0, 0, 1, "FOO"]),
    pd.Series([0, 0, 0, 1, "FOO"]),
    pd.DataFrame([0, 0, 0, 1, "FOO"]),
    "Up the arsenal",
]


@pytest.mark.parametrize("y_correct", INPUT_CORRECT)
def test_check_y_correct(y_correct):
    """Test the _check_y method with correct input."""
    assert BaseSeriesEstimator._check_y(None, y_correct) is None


@pytest.mark.parametrize("y_wrong", INPUT_WRONG)
def test_check_y_wrong(y_wrong):
    """Test the _check_y method with incorrect input."""
    with pytest.raises(ValueError, match="Error in input type for y"):
        BaseSeriesEstimator._check_y(None, y_wrong)
