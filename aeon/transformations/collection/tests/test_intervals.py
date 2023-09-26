# -*- coding: utf-8 -*-
"""Interval extraction test code."""

from aeon.transformations.collection import Catch22, SevenNumberSummaryTransformer
from aeon.transformations.collection.random_intervals import RandomIntervals
from aeon.transformations.collection.supervised_intervals import SupervisedIntervals
from aeon.utils._testing.collection import make_3d_test_data
from aeon.utils.numba.stats import row_mean, row_median


def test_interval_prune():
    X, y = make_3d_test_data(random_state=0, n_channels=2, n_timepoints=10)

    rit = RandomIntervals(
        features=[row_mean, row_median],
        n_intervals=10,
        random_state=0,
    )
    X_t = rit.fit_transform(X, y)

    assert X_t.shape == (10, 16)
    assert rit.transform(X).shape == (10, 16)


def test_random_interval_transformer():
    X, y = make_3d_test_data(random_state=0, n_channels=2, n_timepoints=10)

    rit = RandomIntervals(
        features=SevenNumberSummaryTransformer(),
        n_intervals=5,
        random_state=2,
    )
    X_t = rit.fit_transform(X, y)

    assert X_t.shape == (10, 35)
    assert rit.transform(X).shape == (10, 35)


def test_supervised_transformers():
    X, y = make_3d_test_data(random_state=0)

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

    assert X_t.shape == (X.shape[0], 7)
