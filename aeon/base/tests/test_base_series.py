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


UNIVARIATE = {
    "ndarray": np.random.random(size=(20)),
    "Series": pd.Series(np.random.random(size=(20))),
    "DataFrame": pd.DataFrame(np.random.random(size=(20))),
}
MULTIVARIATE = {
    "ndarray": np.random.random(size=(5, 20)),
    "Series": pd.Series(np.random.random(size=(20))),
    "DataFrame": pd.DataFrame(np.random.random(size=(5, 20))),
}
VALID_TYPES = [
    "ndarray",
    "Series",
    "DataFrame",
]


def test_univariate_convert_X():
    # np.ndarray inner, no multivariate. Univariate are always series,
    # not dataframes, axis is ignored
    dummy1 = BaseSeriesEstimator()
    for str in ["ndarray", "Series"]:  # Inner type, DataFrame not allowed for
        # univariate
        for str2 in VALID_TYPES:  # input type
            X = UNIVARIATE[str2]
            dummy1.set_tags(**{"X_inner_type": str})
            X2 = dummy1._convert_X(X, axis=0)
            assert type(X2).__name__ == str
            X2 = dummy1._convert_X(X, axis=1)
            assert len(X) == len(X2)


def test_multivariate_convert_X():
    # np.ndarray inner multivariate capable. Multivariate are always DataFrames,
    # not Series. 1D np.ndarray converted to 2D. Capability is tested in check_X,
    # not here
    dummy1 = BaseSeriesEstimator(axis=0)
    for str in ["ndarray", "DataFrame"]:  # Inner type, Series not allowed for
        # Multivariate
        for str2 in VALID_TYPES:  # input type
            dummy1.set_tags(**{"X_inner_type": str, "capability:multivariate": True})
            X = MULTIVARIATE[str2]
            X2 = dummy1._convert_X(X, axis=0)
            assert type(X2).__name__ == str
            shape1 = X2.shape
            X2 = dummy1._convert_X(X, axis=1)
            shape2 = X2.shape
            assert shape1[0] == shape2[1] and shape1[1] == shape2[0]


@pytest.mark.parametrize("input_type", VALID_TYPES)
@pytest.mark.parametrize("inner_type", VALID_TYPES)
def test__preprocess_series(input_type, inner_type):
    dummy1 = BaseSeriesEstimator(axis=0)
    if inner_type != "DataFrame":
        dummy1.set_tags(**{"X_inner_type": inner_type})
        X = UNIVARIATE[input_type]
        X2 = dummy1._preprocess_series(X)
        assert type(X2).__name__ == inner_type
    if inner_type != "Series":
        dummy1.set_tags(**{"X_inner_type": inner_type, "capability:multivariate": True})
        X = UNIVARIATE[input_type]
        X2 = dummy1._preprocess_series(X)
        assert type(X2).__name__ == inner_type
