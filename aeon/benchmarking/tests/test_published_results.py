"""Test published result loaders."""

import pytest

from aeon.benchmarking.published_results import (
    get_bake_off_2017_results,
    get_bake_off_2021_results,
    get_bake_off_2023_results,
)
from aeon.benchmarking.results_loaders import CONNECTION_ERRORS
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it relies on external website.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_bake_off_2017_results():
    """Test original bake off results."""
    default_results = get_bake_off_2017_results()
    assert default_results.shape == (85, 25)
    assert default_results[0][0] == 0.6649616368286445
    assert default_results[84][24] == 0.853
    average_results = get_bake_off_2017_results(default_only=False)
    assert average_results.shape == (85, 25)
    assert average_results[0][0] == 0.6575447570332481
    assert average_results[84][24] == 0.8578933333100001


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it relies on external website.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_bake_off_2020_results():
    """Test multivariate bake off results."""
    default_results = get_bake_off_2021_results()
    assert default_results.shape == (26, 11)
    assert default_results[0][0] == 0.99
    assert default_results[25][10] == 0.775
    average_results = get_bake_off_2021_results(default_only=False)
    assert average_results.shape == (26, 11)
    assert average_results[0][0] == 0.9755555555555556
    assert average_results[25][10] == 0.8505208333333333


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it relies on external website.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_bake_off_2023_results():
    """Test bake off redux results."""
    default_results = get_bake_off_2023_results()
    assert default_results.shape == (112, 34)
    assert default_results[0][0] == 0.7774936061381074
    assert default_results[111][32] == 0.9504373177842566
    average_results = get_bake_off_2023_results(default_only=False)
    assert average_results.shape == (112, 34)
    assert average_results[0][0] == 0.7692242114236999
    assert average_results[111][32] == 0.9428571428571431
