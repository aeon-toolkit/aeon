# -*- coding: utf-8 -*-
"""Test functions for data input and output."""

__author__ = ["SebasKoel", "Emiliathewolf", "TonyBagnall", "jasonlines", "achieveordie"]

__all__ = []

import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import aeon
from aeon.datasets import (
    load_from_arff_file,
    load_from_long_to_dataframe,
    load_from_tsf_file,
    load_from_tsfile,
    load_from_tsv_file,
    load_tsf_to_dataframe,
    load_uschange,
)
from aeon.datasets._data_generators import (
    _convert_tsf_to_hierarchical,
    make_example_long_table,
)
from aeon.datasets._data_loaders import DIRNAME, MODULE, _load_saved_dataset
from aeon.datatypes import check_is_mtype


@pytest.mark.parametrize("return_X_y", [True, False])
@pytest.mark.parametrize("return_type", ["nested_univ", "numpy3D", "numpy2D"])
def test_load_provided_dataset(return_X_y, return_type):
    """Test function to check for proper loading.

    Check all possibilities of return_X_y and return_type.
    """
    if return_X_y:
        X, y = _load_saved_dataset("UnitTest", "TRAIN", return_X_y, return_type)
        assert isinstance(y, np.ndarray)
    else:
        X = _load_saved_dataset("UnitTest", "TRAIN", return_X_y, return_type)
    if not return_X_y or return_type == "nested_univ":
        assert isinstance(X, pd.DataFrame)
    elif return_type == "numpy3D":
        assert isinstance(X, np.ndarray) and X.ndim == 3
    elif return_type == "numpy2D":
        assert isinstance(X, np.ndarray) and X.ndim == 2

    # Check whether object is same mtype or not, via bool


def test_load_from_tsfile():
    """Test function for loading TS formats.

    Test
    1. Univariate equal length (UnitTest) returns 3D numpy X, 1D numpy y
    2. Multivariate equal length (BasicMotions) returns 3D numpy X, 1D numpy y
    3. Univariate and multivariate unequal length (PLAID) return X as list of numpy
    """
    data_path = MODULE + "/" + DIRNAME + "/UnitTest/UnitTest_TRAIN.ts"
    # Test 1.1: load univariate equal length (UnitTest), should return 2D array and 1D
    # array, test first and last data
    # Test 1.2: Load a problem without y values (UnitTest),  test first and last data.
    X, y = load_from_tsfile(data_path, return_meta_data=False)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert X.shape == (20, 1, 24) and y.shape == (20,)
    assert X[0][0][0] == 573.0
    X2, y = load_from_tsfile(data_path, return_meta_data=False)
    assert isinstance(X2, np.ndarray)
    assert X2.ndim == 3
    assert X2.shape == (20, 1, 24)
    assert X2[0][0][0] == 573.0

    # Test 2: load multivare equal length (BasicMotions), should return 3D array and 1D
    # array, test first and last data.
    data_path = MODULE + "/data/BasicMotions/BasicMotions_TRAIN.ts"
    X, y = load_from_tsfile(data_path, return_meta_data=False)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (40, 6, 100) and y.shape == (40,)
    assert X[1][2][3] == -1.898794
    X, y = load_from_tsfile(data_path, return_meta_data=False)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert X.shape == (40, 6, 100) and y.shape == (40,)
    assert X[1][2][3] == -1.898794

    # Test 3.1: load univariate unequal length (PLAID), should return a one column
    # dataframe,
    data_path = MODULE + "/data/PLAID/PLAID_TRAIN.ts"
    X, y = load_from_tsfile(full_file_path_and_name=data_path, return_meta_data=False)
    assert isinstance(X, list) and isinstance(y, np.ndarray)
    assert len(X) == 537 and y.shape == (537,)
    # Test 3.2: load multivariate unequal length (JapaneseVowels), should return a X
    # columns dataframe,
    data_path = MODULE + "/data/JapaneseVowels/JapaneseVowels_TRAIN.ts"
    X, y = load_from_tsfile(full_file_path_and_name=data_path, return_meta_data=False)
    assert isinstance(X, list) and isinstance(y, np.ndarray)
    assert len(X) == 270 and y.shape == (270,)


_CHECKS = {
    "uschange": {
        "columns": ["Income", "Production", "Savings", "Unemployment"],
        "len_y": 187,
        "len_X": 187,
        "data_types_X": {
            "Income": "float64",
            "Production": "float64",
            "Savings": "float64",
            "Unemployment": "float64",
        },
        "data_type_y": "float64",
        "data": load_uschange(),
    },
}


@pytest.mark.parametrize("dataset", sorted(_CHECKS.keys()))
def test_forecasting_data_loaders(dataset):
    """
    Assert if datasets are loaded correctly.

    dataset: dictionary with values to assert against should contain:
        'columns' : list with column names in correct order,
        'len_y'   : lenght of the y series (int),
        'len_X'   : lenght of the X series/dataframe (int),
        'data_types_X' : dictionary with column name keys and dtype as value,
        'data_type_y'  : dtype if y column (string)
        'data'    : tuple with y series and X series/dataframe if one is not
                    applicable fill with None value,
    """
    checks = _CHECKS[dataset]
    y = checks["data"][0]
    X = checks["data"][1]

    if y is not None:
        assert isinstance(y, pd.Series)
        assert len(y) == checks["len_y"]
        assert y.dtype == checks["data_type_y"]

    if X is not None:
        if len(checks["data_types_X"]) > 1:
            assert isinstance(X, pd.DataFrame)
        else:
            assert isinstance(X, pd.Series)

        assert X.columns.values.tolist() == checks["columns"]

        for col, dt in checks["data_types_X"].items():
            assert X[col].dtype == dt

        assert len(X) == checks["len_X"]


def test_load_from_long_to_dataframe(tmpdir):
    """Test for loading from long to dataframe."""
    # create and save a example long-format file to csv
    test_dataframe = make_example_long_table()
    dataframe_path = tmpdir.join("data.csv")
    test_dataframe.to_csv(dataframe_path, index=False)
    # load and convert the csv to aeon-formatted data
    nested_dataframe = load_from_long_to_dataframe(dataframe_path)
    assert isinstance(nested_dataframe, pd.DataFrame)


def test_load_from_long_incorrect_format(tmpdir):
    """Test for loading from long with incorrect format."""
    with pytest.raises(ValueError):
        dataframe = make_example_long_table()
        dataframe.drop(dataframe.columns[[3]], axis=1, inplace=True)
        dataframe_path = tmpdir.join("data.csv")
        dataframe.to_csv(dataframe_path, index=False)
        load_from_long_to_dataframe(dataframe_path)


@pytest.mark.parametrize(
    "input_path, return_type, output_df",
    [
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader.tsf",
            "default_tsf",
            pd.DataFrame(
                {
                    "series_name": ["T1", "T2", "T3"],
                    "start_timestamp": [
                        pd.Timestamp(year=1979, month=1, day=1),
                        pd.Timestamp(year=1979, month=1, day=1),
                        pd.Timestamp(year=1973, month=1, day=1),
                    ],
                    "series_value": [
                        [
                            25092.2284,
                            24271.5134,
                            25828.9883,
                            27697.5047,
                            27956.2276,
                            29924.4321,
                            30216.8321,
                        ],
                        [887896.51, 887068.98, 971549.04],
                        [227921, 230995, 183635, 238605, 254186],
                    ],
                }
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader_hierarchical.tsf",
            "pd_multiindex_hier",
            pd.DataFrame(
                data=[
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                    887896.51,
                    887068.98,
                    971549.04,
                    227921,
                    230995,
                    183635,
                    238605,
                    254186,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("G1", "T1", pd.Timestamp(year=1979, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1980, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1981, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1982, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1983, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1984, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1985, month=1, day=1)),
                        ("G1", "T2", pd.Timestamp(year=1979, month=1, day=1)),
                        ("G1", "T2", pd.Timestamp(year=1980, month=1, day=1)),
                        ("G1", "T2", pd.Timestamp(year=1981, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1973, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1974, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1975, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1976, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1977, month=1, day=1)),
                    ],
                    names=["series_group", "series_name", "timestamp"],
                ),
                columns=["series_value"],
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader.tsf",
            "pd-multiindex",
            pd.DataFrame(
                data=[
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                    887896.51,
                    887068.98,
                    971549.04,
                    227921,
                    230995,
                    183635,
                    238605,
                    254186,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("T1", pd.Timestamp(year=1979, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1980, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1981, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1982, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1983, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1984, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1985, month=1, day=1)),
                        ("T2", pd.Timestamp(year=1979, month=1, day=1)),
                        ("T2", pd.Timestamp(year=1980, month=1, day=1)),
                        ("T2", pd.Timestamp(year=1981, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1973, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1974, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1975, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1976, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1977, month=1, day=1)),
                    ],
                    names=["series_name", "timestamp"],
                ),
                columns=["series_value"],
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader_no_start_timestamp.tsf",
            "default_tsf",
            pd.DataFrame(
                {
                    "series_name": ["T1", "T2", "T3"],
                    "series_value": [
                        [
                            25092.2284,
                            24271.5134,
                            25828.9883,
                            27697.5047,
                            27956.2276,
                            29924.4321,
                            30216.8321,
                        ],
                        [887896.51, 887068.98, 971549.04],
                        [227921, 230995, 183635, 238605, 254186],
                    ],
                }
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader_no_start_timestamp.tsf",
            "pd-multiindex",
            pd.DataFrame(
                data=[
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                    887896.51,
                    887068.98,
                    971549.04,
                    227921,
                    230995,
                    183635,
                    238605,
                    254186,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("T1", 0),
                        ("T1", 1),
                        ("T1", 2),
                        ("T1", 3),
                        ("T1", 4),
                        ("T1", 5),
                        ("T1", 6),
                        ("T2", 0),
                        ("T2", 1),
                        ("T2", 2),
                        ("T3", 0),
                        ("T3", 1),
                        ("T3", 2),
                        ("T3", 3),
                        ("T3", 4),
                    ],
                    names=["series_name", "timestamp"],
                ),
                columns=["series_value"],
            ),
        ),
    ],
)
def test_load_tsf_to_dataframe(input_path, return_type, output_df):
    """Test function for loading tsf format."""
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        input_path,
    )

    expected_metadata = {
        "frequency": "yearly",
        "forecast_horizon": 4,
        "contain_missing_values": False,
        "contain_equal_length": False,
    }

    if return_type == "default_tsf":
        df, metadata = load_from_tsf_file(data_path)
    else:
        df, metadata = load_tsf_to_dataframe(data_path, return_type=return_type)

    assert_frame_equal(df, output_df, check_dtype=False)
    assert metadata == expected_metadata
    if return_type != "default_tsf":
        assert check_is_mtype(obj=df, mtype=return_type)


@pytest.mark.parametrize("freq", [None, "YS"])
def test_convert_tsf_to_multiindex(freq):
    input_df = pd.DataFrame(
        {
            "series_name": ["T1", "T2", "T3"],
            "start_timestamp": [
                pd.Timestamp(year=1979, month=1, day=1),
                pd.Timestamp(year=1979, month=1, day=1),
                pd.Timestamp(year=1973, month=1, day=1),
            ],
            "series_value": [
                [
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                ],
                [887896.51, 887068.98, 971549.04],
                [227921, 230995, 183635, 238605, 254186],
            ],
        }
    )

    output_df = pd.DataFrame(
        data=[
            25092.2284,
            24271.5134,
            25828.9883,
            27697.5047,
            27956.2276,
            29924.4321,
            30216.8321,
            887896.51,
            887068.98,
            971549.04,
            227921,
            230995,
            183635,
            238605,
            254186,
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("T1", pd.Timestamp(year=1979, month=1, day=1)),
                ("T1", pd.Timestamp(year=1980, month=1, day=1)),
                ("T1", pd.Timestamp(year=1981, month=1, day=1)),
                ("T1", pd.Timestamp(year=1982, month=1, day=1)),
                ("T1", pd.Timestamp(year=1983, month=1, day=1)),
                ("T1", pd.Timestamp(year=1984, month=1, day=1)),
                ("T1", pd.Timestamp(year=1985, month=1, day=1)),
                ("T2", pd.Timestamp(year=1979, month=1, day=1)),
                ("T2", pd.Timestamp(year=1980, month=1, day=1)),
                ("T2", pd.Timestamp(year=1981, month=1, day=1)),
                ("T3", pd.Timestamp(year=1973, month=1, day=1)),
                ("T3", pd.Timestamp(year=1974, month=1, day=1)),
                ("T3", pd.Timestamp(year=1975, month=1, day=1)),
                ("T3", pd.Timestamp(year=1976, month=1, day=1)),
                ("T3", pd.Timestamp(year=1977, month=1, day=1)),
            ],
            names=["series_name", "timestamp"],
        ),
        columns=["series_value"],
    )

    metadata = {
        "frequency": "yearly",
        "forecast_horizon": 4,
        "contain_missing_values": False,
        "contain_equal_length": False,
    }

    assert_frame_equal(
        output_df,
        _convert_tsf_to_hierarchical(input_df, metadata, freq=freq),
        check_dtype=False,
    )


def test_load_from_ucr_tsv():
    """Test that GunPoint is the same when loaded from .ts and .tsv"""
    X, y = _load_saved_dataset("GunPoint", split="TRAIN")
    data_path = MODULE + "/" + DIRNAME + "/GunPoint/GunPoint_TRAIN.tsv"
    X2, y2 = load_from_tsv_file(data_path)
    y = y.astype(float)
    np.testing.assert_array_almost_equal(X, X2, decimal=4)
    assert np.array_equal(y, y2)


def test_load_from_arff():
    """Test that GunPoint is the same when loaded from .ts and .arff"""
    X, y = _load_saved_dataset("GunPoint", split="TRAIN")
    data_path = MODULE + "/" + DIRNAME + "/GunPoint/GunPoint_TRAIN.arff"
    X2, y2 = load_from_arff_file(data_path)
    np.testing.assert_array_almost_equal(X, X2, decimal=4)
    assert np.array_equal(y, y2)
