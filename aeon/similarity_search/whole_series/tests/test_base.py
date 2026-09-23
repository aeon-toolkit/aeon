"""Test for whole series similarity search base class.

The fit-3D / predict-2D input-shape test lives in
``similarity_search/tests/test_base.py`` (it already exercises both the subsequence
and whole-series mocks), so it is not duplicated here. This module hosts
whole-series-specific base behavior instead.
"""

__maintainer__ = ["baraline"]

import numpy as np
import pytest

from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockWholeSeriesSearch,
)


def test_check_query_length_valid():
    """A query whose length matches the fitted series passes ``_check_query_length``."""
    estimator = MockWholeSeriesSearch()
    X_fit = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=40, return_y=False
    )
    estimator.fit(X_fit)
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=40)
    # Should not raise.
    estimator._check_query_length(query)
    indexes, _ = estimator.predict(query, k=3)
    assert len(indexes) <= 3


def test_check_query_length_invalid():
    """A whole-series query of the wrong length raises via ``_check_query_length``."""
    estimator = MockWholeSeriesSearch()
    X_fit = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=40, return_y=False
    )
    estimator.fit(X_fit)
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=25)
    with pytest.raises(ValueError, match="Expected X to have 40 timepoints"):
        estimator._check_query_length(query)


def test_fit_stores_metadata():
    """Whole-series fit stores the collection metadata."""
    estimator = MockWholeSeriesSearch()
    X_fit = make_example_3d_numpy(
        n_cases=7, n_channels=1, n_timepoints=30, return_y=False
    )
    estimator.fit(X_fit)
    assert estimator.n_cases_ == 7
    assert estimator.n_channels_ == 1
    assert estimator.n_timepoints_ == 30


def test_predict_returns_case_indices():
    """Whole-series predict returns 1D case-index arrays aligned with distances."""
    estimator = MockWholeSeriesSearch()
    X_fit = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=30, return_y=False
    )
    estimator.fit(X_fit)
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=30)
    indexes, distances = estimator.predict(query, k=4)
    assert indexes.ndim == 1
    assert len(indexes) == len(distances)
    assert len(indexes) <= 4
    assert np.all(indexes < estimator.n_cases_)
