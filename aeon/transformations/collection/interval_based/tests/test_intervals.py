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
    ["int16", "int32", "int64", "float32", "float64"],
)
@pytest.mark.parametrize("normalise_for_search", [True, False])
def test_supervised_intervals_preserves_float_precision(dtype, normalise_for_search):
    """Check precision, selected intervals and values agree between fit paths."""
    X, y = make_example_3d_numpy(
        random_state=0,
        n_channels=2,
        n_timepoints=20,
    )
    X = X.astype(dtype)
    assert np.all(np.ptp(X, axis=2) > 0)

    expected_dtype = np.float32 if dtype == "float32" else np.float64

    sit = SupervisedIntervals(
        features=[row_mean, row_median],
        n_intervals=2,
        random_state=0,
        normalise_for_search=normalise_for_search,
    )
    sit.fit(X, y)
    Xt = sit.transform(X)

    sit_fit_transform = SupervisedIntervals(
        features=[row_mean, row_median],
        n_intervals=2,
        random_state=0,
        normalise_for_search=normalise_for_search,
    )
    Xt_fit_transform = sit_fit_transform.fit_transform(X, y)

    assert sit.intervals_ == sit_fit_transform.intervals_
    assert Xt.shape == Xt_fit_transform.shape == (X.shape[0], len(sit.intervals_))
    assert Xt.shape[1] > 0
    assert Xt.dtype == expected_dtype
    assert Xt_fit_transform.dtype == expected_dtype
    rtol = 1e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(Xt, Xt_fit_transform, rtol=rtol, atol=0)
