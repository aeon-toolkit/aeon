"""Test functions for data input and output."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from aeon.utils.conversion._convert_tsf import _convert_tsf_to_hierarchical


@pytest.mark.parametrize("freq", [None, "YS"])
def test_convert_tsf_to_multiindex(freq):
    """Test convert_tsf_to_multiindex method."""
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
