"""Test functions in tsad_datasets.py."""

import pytest

from aeon.datasets.tsad_datasets import (
    multivariate,
    supervised,
    tsad_collections,
    tsad_datasets,
    univariate,
    unsupervised,
)
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
def test_helper_functions():
    """Test helper functions."""
    d = tsad_collections()
    assert isinstance(d, dict)
    d = tsad_datasets()
    assert isinstance(d, list)
    d = univariate()
    assert isinstance(d, list)
    d = multivariate()
    assert isinstance(d, list)
    d = unsupervised()
    assert isinstance(d, list)
    d = supervised()
    assert isinstance(d, list)
