"""Test functions for dataset collections."""

import pytest

from aeon.datasets.dataset_collections import (
    get_available_tsc_datasets,
    get_available_tser_datasets,
    get_available_tsf_datasets,
    get_downloaded_tsf_datasets,
)
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write.",
)
def test_downloaded_tsf_datasets():
    """Test recovering downloaded data sets.

    This will fail if extra datasets are added and the test remains unchanged.
    """
    res = get_downloaded_tsf_datasets()
    assert len(res) == 1
    assert res[0] == "m1_yearly_dataset"
    with pytest.raises(FileNotFoundError):
        res = get_downloaded_tsf_datasets("FOO")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write.",
)
def test_list_available_tsc_datasets():
    """Test recovering lists of available data sets."""
    res = get_available_tsc_datasets()
    assert len(res) == 158
    res = get_available_tsc_datasets("FOO")
    assert not res
    res = get_available_tsc_datasets("Chinatown")
    assert res


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write.",
)
def test_list_available_tser_datasets():
    """Test recovering lists of available data sets."""
    res = get_available_tser_datasets()
    assert len(res) == 63
    res = get_available_tser_datasets(return_list=False)
    assert isinstance(res, set)
    res = get_available_tser_datasets(name="tser_monash")
    assert len(res) == 18
    res = get_available_tser_datasets(return_list=False, name="tser_monash")
    assert isinstance(res, dict)
    assert len(res) == 18
    res = get_available_tser_datasets("FOO")
    assert not res
    res = get_available_tser_datasets("Covid3Month")
    assert res


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write.",
)
def test_list_available_tsf_datasets():
    """Test recovering lists of available data sets."""
    res = get_available_tsf_datasets()
    assert len(res) == 53
    res = get_available_tsf_datasets("FOO")
    assert not res
    res = get_available_tsf_datasets("m1_monthly_dataset")
    assert res
