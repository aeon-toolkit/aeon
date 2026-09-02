"""Interval extraction test code."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.feature_based import Catch22, SevenNumberSummary
from aeon.transformations.collection.interval_based import (
    RandomIntervals,
    SupervisedIntervals,
)
from aeon.utils.numba.stats import row_mean, row_median


def test_interval_prune():
    """Test pruning of intervals by the RandomIntervals transformer."""
    X, y = make_example_3d_numpy(random_state=0, n_channels=2, n_timepoints=10)

    rit = RandomIntervals(
        features=[row_mean, row_median],
        n_intervals=10,
        random_state=0,
    )
    X_t = rit.fit_transform(X, y)

    assert X_t.shape == (10, 16)
    assert rit.transform(X).shape == (10, 16)


def test_random_interval_transformer():
    """Test the RandomIntervals transformer output."""
    X, y = make_example_3d_numpy(random_state=0, n_channels=2, n_timepoints=20)

    rit = RandomIntervals(
        features=SevenNumberSummary(),
        n_intervals=5,
        random_state=0,
    )
    X_t = rit.fit_transform(X, y)

    assert X_t.shape == (10, 35)
    assert rit.transform(X).shape == (10, 35)


def test_supervised_transformers():
    """Test the SupervisedIntervals transformer output."""
    X, y = make_example_3d_numpy(random_state=0)

    sit = SupervisedIntervals(
        features=[
            Catch22(
                features=["DN_HistogramMode_5", "SB_BinaryStats_mean_longstretch1"]
            ),
            row_mean,
        ],
        n_intervals=2,
        random_state=0,
    )
    X_t = sit.fit_transform(X, y)

    assert X_t.shape == (X.shape[0], 8)


@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "float32", "float64"],
)
def test_supervised_intervals_preserves_float_precision(dtype):
    """Test SupervisedIntervals preserves float32 and promotes integer input."""
    X, y = make_example_3d_numpy(
        random_state=0,
        n_channels=1,
        n_timepoints=20,
    )
    X = X.astype(dtype)

    expected_dtype = np.float32 if dtype == "float32" else np.float64

    sit = SupervisedIntervals(
        features=[row_mean],
        n_intervals=2,
        random_state=0,
    )
    sit.fit(X, y)
    Xt = sit.transform(X)

    sit = SupervisedIntervals(
        features=[row_mean],
        n_intervals=2,
        random_state=0,
    )
    Xt_fit_transform = sit.fit_transform(X, y)

    assert Xt.dtype == expected_dtype
    assert Xt_fit_transform.dtype == expected_dtype
