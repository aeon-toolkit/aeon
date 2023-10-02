# -*- coding: utf-8 -*-
"""Test functions for dataset collections."""
import pytest

from aeon.datasets.dataset_collections import (
    list_available_tsc_datasets,
    list_available_tser_datasets,
    list_available_tsf_datasets,
    list_downloaded_tsf_datasets,
)


def test_downloaded_tsf_datasets():
    """Test recovering downloaded data sets.

    This will fail if extra datasets are added and the test remains unchanged.
    """
    res = list_downloaded_tsf_datasets()
    assert len(res) == 1
    assert res[0] == "m1_yearly_dataset"
    with pytest.raises(FileNotFoundError):
        res = list_downloaded_tsf_datasets("FOO")


def test_list_available_tsc_datasets():
    """Test recovering lists of available data sets."""
    res = list_available_tsc_datasets()
    assert len(res) == 161
    res = list_available_tsc_datasets("FOO")
    assert not res
    res = list_available_tsc_datasets("Chinatown")
    assert res


def test_list_available_tser_datasets():
    """Test recovering lists of available data sets."""
    res = list_available_tser_datasets()
    assert len(res) == 19
    res = list_available_tser_datasets("FOO")
    assert not res
    res = list_available_tser_datasets("Covid3Month")
    assert res


def test_list_available_tsf_datasets():
    """Test recovering lists of available data sets."""
    res = list_available_tsf_datasets()
    assert len(res) == 53
    res = list_available_tsf_datasets("FOO")
    assert not res
    res = list_available_tsf_datasets("m1_monthly_dataset")
    assert res
