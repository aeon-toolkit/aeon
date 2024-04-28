"""Small testing config for Time Series Forecasting."""

__maintainer__ = []

import numpy as np
import pandas as pd

from aeon.testing.utils.data_gen import make_series

# We here define the parameter values for unit testing.
TEST_CUTOFFS_INT_LIST = [[21], [3, 7]]
TEST_CUTOFFS_INT_ARR = [np.array([21, 22]), np.array([3, 10])]
# The following timestamps correspond
# to the above integers for `make_series(all_positive=True)`
TEST_CUTOFFS_TIMESTAMP = [
    pd.to_datetime(["2000-01-23"]),
    pd.to_datetime(["2000-01-04", "2000-01-08"]),
]
TEST_CUTOFFS = [*TEST_CUTOFFS_INT_LIST, *TEST_CUTOFFS_INT_ARR, *TEST_CUTOFFS_TIMESTAMP]

TEST_WINDOW_LENGTHS_INT = [1, 5]
TEST_WINDOW_LENGTHS_TIMEDELTA = [pd.Timedelta(1, unit="D")]
TEST_WINDOW_LENGTHS_DATEOFFSET = [pd.offsets.Day(1)]
TEST_WINDOW_LENGTHS = [
    *TEST_WINDOW_LENGTHS_INT,
    *TEST_WINDOW_LENGTHS_TIMEDELTA,
    *TEST_WINDOW_LENGTHS_DATEOFFSET,
]

TEST_INITIAL_WINDOW_INT = [10]
TEST_INITIAL_WINDOW_TIMEDELTA = [pd.Timedelta(7, unit="D")]
TEST_INITIAL_WINDOW_DATEOFFSET = [pd.offsets.Day(10)]
TEST_INITIAL_WINDOW = [
    *TEST_INITIAL_WINDOW_INT,
    *TEST_INITIAL_WINDOW_TIMEDELTA,
    *TEST_INITIAL_WINDOW_DATEOFFSET,
]

TEST_STEP_LENGTHS_INT = [1, 5]
TEST_STEP_LENGTHS_TIMEDELTA = [pd.Timedelta(1, unit="D"), pd.Timedelta(5, unit="D")]
TEST_STEP_LENGTHS_DATEOFFSET = [pd.offsets.Day(1), pd.offsets.Day(5)]
TEST_STEP_LENGTHS = [
    *TEST_STEP_LENGTHS_INT,
    *TEST_STEP_LENGTHS_TIMEDELTA,
    *TEST_STEP_LENGTHS_DATEOFFSET,
]

TEST_OOS_FHS = [1, np.array([2, 5], dtype="int64")]  # out-of-sample
TEST_INS_FHS = [
    -3,  # single in-sample
    np.array([-2, -5], dtype="int64"),  # multiple in-sample
    np.array([-3, 2], dtype="int64"),  # mixed in-sample and out-of-sample
]
TEST_FHS = [*TEST_OOS_FHS, *TEST_INS_FHS]

TEST_OOS_FHS_TIMEDELTA = [
    [pd.Timedelta(2, unit="D"), pd.Timedelta(5, unit="D")],
]  # out-of-sample
TEST_INS_FHS_TIMEDELTA = [
    pd.Timedelta(-3, unit="D"),  # single in-sample
    [pd.Timedelta(-2, unit="D"), pd.Timedelta(-5, unit="D")],  # multiple in-sample
    pd.Timedelta(0, unit="D"),  # last training point
    [
        pd.Timedelta(-3, unit="D"),
        pd.Timedelta(2, unit="D"),
    ],  # mixed in-sample and out-of-sample
]
TEST_FHS_TIMEDELTA = [*TEST_OOS_FHS_TIMEDELTA, *TEST_INS_FHS_TIMEDELTA]

TEST_SPS = [3]
TEST_ALPHAS = [0.1]
TEST_YS = [make_series(all_positive=True)]
TEST_N_ITERS = [1]

# We currently support the following combinations of index and forecasting horizon types
VALID_INDEX_FH_COMBINATIONS = [
    # index type, fh type, is_relative
    ("int", "int", True),
    ("datetime", "datetime", False),
]

INDEX_TYPE_LOOKUP = {
    "int": pd.Index,
    "datetime": pd.DatetimeIndex,
}
