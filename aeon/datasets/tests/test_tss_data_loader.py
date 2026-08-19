"""Test segmentation dataset loaders."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aeon.datasets import (
    load_human_activity_segmentation_datasets,
    load_time_series_segmentation_benchmark,
)
from aeon.datasets.tests.test_data_loaders import CONNECTION_ERRORS
from aeon.segmentation import ClaSPSegmenter
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of slow read.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_tssb(mocker):
    """Test load time series segmentation benchmark."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mocker.patch("aeon.datasets._tss_data_loaders._DATA_FOLDER", tmp)

        # test download
        X, y = load_time_series_segmentation_benchmark()

        assert isinstance(X, list)
        assert all(isinstance(ts, np.ndarray) for ts in X)
        assert all(ts.ndim == 1 for ts in X)
        assert len(X) == 75

        assert isinstance(y, list)
        assert all(isinstance(cps, np.ndarray) for cps in y)
        assert all(cps.ndim == 1 for cps in y)
        assert len(y) == 75

        # test load + meta data
        X, y, metadata = load_time_series_segmentation_benchmark(return_metadata=True)

        assert isinstance(metadata, list)
        assert len(y) == 75

        # test that segmentation works
        ts, cps, _, window_size = X[0], y[0], *metadata[0]
        clasp = ClaSPSegmenter(period_length=window_size, n_cps=cps.shape[0])
        found_cps = clasp.fit_predict(ts)
        assert cps.shape[0] == found_cps.shape[0]


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of slow read.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_has_datasets(mocker):
    """Test load human activity segmentation data sets."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mocker.patch("aeon.datasets._tss_data_loaders._DATA_FOLDER", tmp)

        # test download
        X, y = load_human_activity_segmentation_datasets()

        assert isinstance(X, list)
        assert all(isinstance(ts, np.ndarray) for ts in X)
        assert all(ts.ndim == 2 for ts in X)
        assert len(X) == 250

        assert isinstance(y, list)
        assert all(isinstance(cps, np.ndarray) for cps in y)
        assert all(cps.ndim == 1 for cps in y)
        assert len(y) == 250

        # test load + meta data
        X, y, metadata = load_human_activity_segmentation_datasets(return_metadata=True)

        assert isinstance(metadata, list)
        assert len(y) == 250

        # test that segmentation works
        ts, cps, sample_rate = X[0], y[0], 50
        clasp = ClaSPSegmenter(period_length=sample_rate, n_cps=cps.shape[0])
        found_cps = clasp.fit_predict(ts[:, 0])
        assert cps.shape[0] == found_cps.shape[0]
