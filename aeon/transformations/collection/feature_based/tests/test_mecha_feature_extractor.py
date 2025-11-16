"""Tests for the Mecha feature extractor utilities."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.feature_based._mecha_feature_extractor import (
    bidirect_dilation_mapping,
    bidirect_interleaving_mapping,
    dilated_fres_extract,
    interleaved_fres_extract,
    series_transform,
    td,
)

N_TIMEPOINTS = 50
MAX_RATE = 8
N_CASES = 5
N_CHANNELS = 2

TSFRESH_FEATURES_PER_CHANNEL = 39

EXPECTED_DILATED_FEATURES = 7770

EXPECTED_INTERLEAVED_FEATURES = 6216


@pytest.fixture
def example_3d_data():
    """Return a simple 3D NumPy array for testing."""
    return make_example_3d_numpy(
        n_cases=N_CASES,
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        random_state=42,
        return_y=False,
    )


@pytest.fixture
def example_series():
    """Return a single time series (1D array) for TD testing."""
    return np.linspace(0, N_TIMEPOINTS - 1, N_TIMEPOINTS)


def test_td_output_length(example_series):
    """Test TD output length is T-1."""
    series = example_series
    h = 1 / len(series)
    dSignal = td(series, k=3, h=h)
    assert len(dSignal) == len(series) - 1


def test_series_transform_output_shape(example_3d_data):
    """Test series_transform output shape is (n_cases, n_channels, n_timepoints - 1)."""
    X = example_3d_data
    X_transformed = series_transform(X, k1=2.0)

    n_cases, n_channels, n_timepoints = X.shape

    assert X_transformed.shape == (n_cases, n_channels, n_timepoints - 1)


def test_dilation_mapping_output(example_3d_data):
    """Test dilation mapping returns correct number of views and length."""
    X = example_3d_data
    indexList = bidirect_dilation_mapping(X, max_rate=MAX_RATE)
    n_timepoints = X.shape[2]
    assert indexList.shape[0] == 4
    assert indexList.shape[1] == n_timepoints


def test_interleaving_mapping_output(example_3d_data):
    """Test interleaving mapping returns correct number of views and length."""
    X = example_3d_data
    indexList = bidirect_interleaving_mapping(X, max_rate=MAX_RATE)
    n_timepoints = X.shape[2]
    assert indexList.shape[0] == 4
    assert indexList.shape[1] == n_timepoints


def test_dilated_fres_extract_output_shape(example_3d_data):
    """Test dilated extraction output shape (including original series)."""
    X = example_3d_data
    features = dilated_fres_extract(X, max_rate=MAX_RATE, basic_extractor="TSFresh")
    assert features.shape[0] == X.shape[0]
    assert features.shape[1] == EXPECTED_DILATED_FEATURES


def test_dilated_fres_extract_catch22_output_shape(example_3d_data):
    """Test dilated extraction output shape using Catch22."""
    X = example_3d_data
    C22_FEATURES_PER_CHANNEL = 22
    expected_num_views = 1 + 4
    expected_num_features = expected_num_views * N_CHANNELS * C22_FEATURES_PER_CHANNEL
    features = dilated_fres_extract(X, max_rate=MAX_RATE, basic_extractor="Catch22")
    assert features.shape[0] == X.shape[0]
    assert features.shape[1] == expected_num_features


def test_interleaved_fres_extract_output_shape(example_3d_data):
    """Test interleaved extraction output shape (only shuffled views)."""
    X = example_3d_data
    features = interleaved_fres_extract(X, max_rate=MAX_RATE, basic_extractor="TSFresh")
    assert features.shape[0] == X.shape[0]
    assert features.shape[1] == EXPECTED_INTERLEAVED_FEATURES
