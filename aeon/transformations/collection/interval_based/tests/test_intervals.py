"""Interval extraction test code."""

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
