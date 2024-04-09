"""Tests for base series estimator."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from aeon.base import BaseSeriesEstimator

UNIVARIATE = {
    "np.ndarray": np.random.random(size=(20)),
    "pd.Series": pd.Series(np.random.random(size=(20))),
    "pd.DataFrame": pd.DataFrame(np.random.random(size=(1, 20))),
}
MULTIVARIATE = {
    "np.ndarray": np.random.random(size=(5, 20)),
    "pd.DataFrame": pd.DataFrame(np.random.random(size=(5, 20))),
}

UNIVARIATE_MISSING = {
    "np.ndarray": UNIVARIATE["np.ndarray"].copy(),
    "pd.Series": UNIVARIATE["pd.Series"].copy(),
    "pd.DataFrame": UNIVARIATE["pd.DataFrame"].copy(),
}
UNIVARIATE_MISSING["np.ndarray"][np.random.randint(20)] = np.NAN
UNIVARIATE_MISSING["np.ndarray"][np.random.randint(20)] = np.NAN
UNIVARIATE_MISSING["pd.Series"][np.random.randint(20)] = np.NAN
UNIVARIATE_MISSING["pd.Series"][np.random.randint(20)] = np.NAN
UNIVARIATE_MISSING["pd.DataFrame"].iloc[0, np.random.randint(20)] = np.NAN
UNIVARIATE_MISSING["pd.DataFrame"].iloc[0, np.random.randint(20)] = np.NAN

MULTIVARIATE_MISSING = {
    "np.ndarray": MULTIVARIATE["np.ndarray"].copy(),
    "pd.DataFrame": MULTIVARIATE["pd.DataFrame"].copy(),
}
MULTIVARIATE_MISSING["np.ndarray"][np.random.randint(5)][np.random.randint(20)] = np.NAN
MULTIVARIATE_MISSING["np.ndarray"][np.random.randint(5)][np.random.randint(20)] = np.NAN
MULTIVARIATE_MISSING["pd.DataFrame"].iloc[
    np.random.randint(5), np.random.randint(20)
] = np.NAN
MULTIVARIATE_MISSING["pd.DataFrame"].iloc[
    np.random.randint(5), np.random.randint(20)
] = np.NAN


VALID_TYPES = [
    "np.ndarray",
    "pd.Series",
    "pd.DataFrame",
]


def test_check_X():
    """Test if capabilities correctly tested in _check_X."""
    dummy = BaseSeriesEstimator(axis=1)

    # check basic univariate input
    meta = dummy._check_X(UNIVARIATE["np.ndarray"], axis=1)
    meta2 = dummy._check_X(UNIVARIATE["pd.Series"], axis=1)
    meta3 = dummy._check_X(UNIVARIATE["pd.DataFrame"], axis=1)
    assert meta == meta2 == meta3
    assert not meta["multivariate"]
    assert meta["n_channels"] == 1
    assert not meta["missing_values"]

    # DataFrames are always 2d
    meta = dummy._check_X(UNIVARIATE["pd.DataFrame"].T, axis=0)
    assert not meta["multivariate"]
    assert meta["n_channels"] == 1
    assert not meta["missing_values"]

    # check invalid inputs and data types
    invalid_always = np.array(["1", "2", "3", "4", "5"])

    with pytest.raises(ValueError, match=r"Input type of X should be one of"):
        dummy._check_X("String input", axis=1)
    with pytest.raises(ValueError, match=r"dtype for np.ndarray must be float or int"):
        dummy._check_X(invalid_always, axis=1)
    with pytest.raises(ValueError, match=r"pd.Series dtype must be numeric"):
        dummy._check_X(pd.Series(invalid_always), axis=1)
    with pytest.raises(ValueError, match=r"pd.DataFrame dtype must be numeric"):
        dummy._check_X(pd.DataFrame(invalid_always), axis=1)

    # check multivariate capability False
    with pytest.raises(ValueError, match=r"Multivariate data not supported"):
        dummy._check_X(MULTIVARIATE["np.ndarray"], axis=1)
    with pytest.raises(ValueError, match=r"Multivariate data not supported"):
        dummy._check_X(MULTIVARIATE["pd.DataFrame"], axis=1)
    with pytest.raises(ValueError, match=r"Multivariate data not supported"):
        dummy._check_X(MULTIVARIATE["np.ndarray"].T, axis=0)
    with pytest.raises(ValueError, match=r"Multivariate data not supported"):
        dummy._check_X(MULTIVARIATE["pd.DataFrame"].T, axis=0)

    # check missing value capability False
    dummy.set_tags(**{"capability:multivariate": True})

    with pytest.raises(ValueError, match=r"Missing values not supported"):
        dummy._check_X(UNIVARIATE_MISSING["np.ndarray"], axis=1)
    with pytest.raises(ValueError, match=r"Missing values not supported"):
        dummy._check_X(UNIVARIATE_MISSING["pd.Series"], axis=1)
    with pytest.raises(ValueError, match=r"Missing values not supported"):
        dummy._check_X(UNIVARIATE_MISSING["pd.DataFrame"], axis=1)
    with pytest.raises(ValueError, match=r"Missing values not supported"):
        dummy._check_X(MULTIVARIATE_MISSING["np.ndarray"], axis=1)
    with pytest.raises(ValueError, match=r"Missing values not supported"):
        dummy._check_X(MULTIVARIATE_MISSING["pd.DataFrame"], axis=1)

    # check multivariate capable
    meta = dummy._check_X(MULTIVARIATE["np.ndarray"], axis=1)
    meta2 = dummy._check_X(MULTIVARIATE["pd.DataFrame"], axis=1)
    meta3 = dummy._check_X(MULTIVARIATE["np.ndarray"].T, axis=0)
    meta4 = dummy._check_X(MULTIVARIATE["pd.DataFrame"].T, axis=0)
    assert meta == meta2 == meta3 == meta4
    assert meta["multivariate"]
    assert meta["n_channels"] == 5
    assert not meta["missing_values"]

    # check missing value capable
    dummy.set_tags(**{"capability:missing_values": True})

    meta = dummy._check_X(UNIVARIATE_MISSING["np.ndarray"], axis=1)
    meta2 = dummy._check_X(UNIVARIATE_MISSING["pd.Series"], axis=1)
    meta3 = dummy._check_X(UNIVARIATE_MISSING["pd.DataFrame"], axis=1)
    meta4 = dummy._check_X(MULTIVARIATE_MISSING["np.ndarray"], axis=1)
    meta5 = dummy._check_X(MULTIVARIATE_MISSING["pd.DataFrame"], axis=1)
    assert meta == meta2 == meta3
    assert meta4 == meta5
    assert meta["missing_values"] and meta4["missing_values"]
    assert not meta["multivariate"]
    assert meta["n_channels"] == 1
    assert meta4["multivariate"]
    assert meta4["n_channels"] == 5

    # check univariate capability False
    dummy.set_tags(**{"capability:univariate": False})

    with pytest.raises(ValueError, match=r"Univariate data not supported"):
        dummy._check_X(UNIVARIATE["np.ndarray"], axis=1)
    with pytest.raises(ValueError, match=r"Univariate data not supported"):
        dummy._check_X(UNIVARIATE["pd.Series"], axis=1)
    with pytest.raises(ValueError, match=r"Univariate data not supported"):
        dummy._check_X(UNIVARIATE["pd.DataFrame"], axis=1)
    with pytest.raises(ValueError, match=r"Univariate data not supported"):
        dummy._check_X(UNIVARIATE["pd.DataFrame"].T, axis=0)


@pytest.mark.parametrize("input_type", VALID_TYPES)
def test_convert_X_ndarray_inner(input_type):
    """Test _convert_X on with np.ndarray inner type."""
    dummy = BaseSeriesEstimator(axis=1)
    dummy.set_tags(**{"X_inner_type": "np.ndarray"})
    # test univariate
    X = UNIVARIATE[input_type]

    X2 = dummy._convert_X(X, axis=1)
    assert type(X2).__name__ == "ndarray"
    assert X2.shape == (1, 20)
    assert X.shape[-1] == X2.shape[1]

    X3 = dummy._convert_X(X.T, axis=0)
    assert type(X3).__name__ == "ndarray"
    assert X3.shape == (1, 20)
    assert_equal(X2, X3)

    # test multivariate
    if input_type != "pd.Series":
        dummy.set_tags(**{"capability:multivariate": True})
        X = MULTIVARIATE[input_type]

        X2 = dummy._convert_X(X, axis=1)
        assert type(X2).__name__ == "ndarray"
        assert X2.shape == (5, 20)
        assert X.shape == X2.shape

        X3 = dummy._convert_X(X.T, axis=0)
        assert type(X3).__name__ == "ndarray"
        assert X3.shape == (5, 20)
        assert_equal(X2, X3)


@pytest.mark.parametrize("input_type", VALID_TYPES)
def test_convert_X_series_inner(input_type):
    """Test _convert_X on with pd.Series inner type."""
    dummy = BaseSeriesEstimator(axis=1)
    dummy.set_tags(**{"X_inner_type": "pd.Series"})
    # test univariate
    X = UNIVARIATE[input_type]

    X2 = dummy._convert_X(X, axis=1)
    assert type(X2).__name__ == "Series"
    assert X2.shape == (20,)
    assert X.shape[-1] == X2.shape[0]

    X3 = dummy._convert_X(X.T, axis=0)
    assert type(X3).__name__ == "Series"
    assert X3.shape == (20,)
    assert X2.equals(X3)

    # test multivariate, should raise error as pd.Series cannot be multivariate
    if input_type != "pd.Series":
        X = MULTIVARIATE[input_type]

        with pytest.raises(ValueError, match="multivariate data. Found 5 channels"):
            dummy._convert_X(X, axis=1)

        dummy.set_tags(**{"capability:multivariate": True})

        with pytest.raises(ValueError, match="for multivariate capable estimators"):
            dummy._convert_X(X, axis=1)
        with pytest.raises(ValueError, match="for multivariate capable estimators"):
            dummy._convert_X(X.T, axis=0)


@pytest.mark.parametrize("input_type", VALID_TYPES)
def test_convert_X_dataframe_inner(input_type):
    """Test _convert_X on with pd.DataFrame inner type."""
    dummy = BaseSeriesEstimator(axis=1)
    dummy.set_tags(**{"X_inner_type": "pd.DataFrame"})
    # test univariate
    X = UNIVARIATE[input_type]

    X2 = dummy._convert_X(X, axis=1)
    assert type(X2).__name__ == "DataFrame"
    assert X2.shape == (1, 20)
    assert X.shape[-1] == X2.shape[1]

    X3 = dummy._convert_X(X.T, axis=0)
    assert type(X3).__name__ == "DataFrame"
    assert X3.shape == (1, 20)
    assert X2.equals(X3)

    # test multivariate
    if input_type != "pd.Series":
        dummy.set_tags(**{"capability:multivariate": True})
        X = MULTIVARIATE[input_type]

        X2 = dummy._convert_X(X, axis=1)
        assert type(X2).__name__ == "DataFrame"
        assert X2.shape == (5, 20)
        assert X.shape == X2.shape

        X3 = dummy._convert_X(X.T, axis=0)
        assert type(X3).__name__ == "DataFrame"
        assert X3.shape == (5, 20)
        assert X2.equals(X3)


@pytest.mark.parametrize("input_type", VALID_TYPES)
@pytest.mark.parametrize("inner_type", VALID_TYPES)
def test_preprocess_series(input_type, inner_type):
    """Test _preprocess_series for different input and inner types."""
    dummy = BaseSeriesEstimator(axis=1)
    dummy.set_tags(**{"X_inner_type": inner_type})
    inner_name = inner_type.split(".")[1]
    # test univariate
    X = UNIVARIATE[input_type]

    X2 = dummy._preprocess_series(X, axis=1, overwrite_metadata=True)
    assert type(X2).__name__ == inner_name
    assert X.shape[-1] == X2.shape[-1]
    assert not dummy.metadata_["multivariate"]
    assert dummy.metadata_["n_channels"] == 1
    assert not dummy.metadata_["missing_values"]

    X3 = dummy._preprocess_series(X.T, axis=0, overwrite_metadata=True)
    assert type(X3).__name__ == inner_name
    assert X.shape[-1] == X3.shape[-1]
    assert not dummy.metadata_["multivariate"]
    assert dummy.metadata_["n_channels"] == 1
    assert not dummy.metadata_["missing_values"]

    # test multivariate, excludes pd.series input type as it cannot be multivariate
    if input_type != "pd.Series" and inner_type != "pd.Series":
        dummy.set_tags(**{"capability:multivariate": True})
        X = MULTIVARIATE[input_type]

        X2 = dummy._preprocess_series(X, axis=1, overwrite_metadata=True)
        assert type(X2).__name__ == inner_name
        assert X2.shape == (5, 20)
        assert X.shape == X2.shape
        assert dummy.metadata_["multivariate"]
        assert dummy.metadata_["n_channels"] == 5
        assert not dummy.metadata_["missing_values"]

        X3 = dummy._preprocess_series(X.T, axis=0, overwrite_metadata=True)
        assert type(X3).__name__ == inner_name
        assert X3.shape == (5, 20)
        assert X.shape == X3.shape
        assert dummy.metadata_["multivariate"]
        assert dummy.metadata_["n_channels"] == 5
        assert not dummy.metadata_["missing_values"]
    elif input_type != "pd.Series":
        dummy.set_tags(**{"capability:multivariate": True})
        X = MULTIVARIATE[input_type]

        with pytest.raises(ValueError, match="Cannot convert to pd.Series"):
            dummy._preprocess_series(X, axis=1, overwrite_metadata=True)
