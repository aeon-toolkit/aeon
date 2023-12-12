"""Test functions for dataset collections dataframe loaders."""
import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import aeon
from aeon.datasets import (
    load_from_arff_to_dataframe,
    load_from_tsfile_to_dataframe,
    load_from_ucr_tsv_to_dataframe,
    load_tsf_to_dataframe,
)
from aeon.datasets._data_generators import _convert_tsf_to_hierarchical
from aeon.tests.test_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_tsfile_to_dataframe():
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTest_TRAIN.ts",
    )

    data, y = load_from_tsfile_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (20, 1)
    assert len(y) == 20


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_arff_to_dataframe():
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/GunPoint/GunPoint_TRAIN.arff",
    )

    data, y = load_from_arff_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (50, 1)
    assert len(y) == 50


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_ucr_tsv_to_dataframe():
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/GunPoint/GunPoint_TRAIN.tsv",
    )

    data, y = load_from_ucr_tsv_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (50, 1)
    assert len(y) == 50


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_tsf_to_dataframe_simple():
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/m1_yearly_dataset/m1_yearly_dataset.tsf",
    )

    data, meta = load_tsf_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (181, 3)
    assert len(meta) == 4
    data, meta = load_tsf_to_dataframe(data_path, return_type="pd-multiindex")
    assert type(data) is pd.DataFrame
    assert data.shape == (4515, 1)
    assert len(meta) == 4


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
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


@pytest.mark.parametrize(
    "input_path, return_type, output_df",
    [
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader.tsf",
            "tsf_default",
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
            "tsf_default",
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

    df, metadata = load_tsf_to_dataframe(data_path, return_type=return_type)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(metadata, dict)
    assert_frame_equal(df, output_df, check_dtype=False)
    assert metadata == expected_metadata
    if return_type == "tsf_default":
        assert isinstance(df, pd.DataFrame)
    elif return_type == "pd-multiindex":
        assert isinstance(df.index, pd.MultiIndex)
    elif return_type == "pd_multiindex_hier":
        assert df.index.nlevels > 1
