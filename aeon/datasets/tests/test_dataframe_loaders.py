# -*- coding: utf-8 -*-
"""Test functions for dataset collections dataframe loaders."""
import os

import pandas as pd

import aeon
from aeon.datasets import load_from_tsfile_to_dataframe


def test_load_from_tsfile_to_dataframe():
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTestTimeStamps_TRAIN.ts",
    )

    data, y = load_from_tsfile_to_dataframe(data_path)
    assert type(data) == pd.DataFrame
    assert data.shape == (4, 1)
    assert len(y) == 4
