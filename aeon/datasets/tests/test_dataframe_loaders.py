"""Test functions for dataset collections dataframe loaders."""

import os

import pandas as pd
import pytest

import aeon
from aeon.datasets import (
    load_from_arff_to_dataframe,
    load_from_tsfile_to_dataframe,
    load_from_ucr_tsv_to_dataframe,
)
from aeon.testing.test_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_tsfile_to_dataframe():
    """Test function to check functionality of load_from_tsfile_to_dataframe."""
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTest_TRAIN.ts",
    )

    data, y = load_from_tsfile_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (20, 1)
    assert len(y) == 20


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_arff_to_dataframe():
    """Test function to check functionality of load_from_arff_to_dataframe."""
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/GunPoint/GunPoint_TRAIN.arff",
    )

    data, y = load_from_arff_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (50, 1)
    assert len(y) == 50


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_ucr_tsv_to_dataframe():
    """Test function to check functionality of load_from_ucr_tsv_to_dataframe."""
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/GunPoint/GunPoint_TRAIN.tsv",
    )

    data, y = load_from_ucr_tsv_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (50, 1)
    assert len(y) == 50
