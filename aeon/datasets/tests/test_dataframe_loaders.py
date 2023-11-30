"""Test functions for dataset collections dataframe loaders."""
import os

import pandas as pd
import pytest

import aeon
from aeon.datasets import load_from_tsfile_to_dataframe
from aeon.tests.test_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_tsfile_to_dataframe():
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTestTimeStamps_TRAIN.ts",
    )

    data, y = load_from_tsfile_to_dataframe(data_path)
    assert type(data) is pd.DataFrame
    assert data.shape == (4, 1)
    assert len(y) == 4
