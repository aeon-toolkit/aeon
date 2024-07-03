"""Test segmentation dataset loaders."""

import tempfile
from pathlib import Path

import numpy as np

from aeon.datasets import (
    load_human_activity_segmentation_datasets,
    load_time_series_segmentation_benchmark,
)


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
