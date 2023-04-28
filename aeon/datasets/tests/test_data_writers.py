# -*- coding: utf-8 -*-
import shutil

import numpy as np
import pytest

"""Test functions for data writing."""
from aeon.datasets import (
    _load_provided_dataset,
    load_from_tsfile,
    write_collection_to_tsfile,
)


@pytest.mark.parametrize("dataset_name", ["UnitTest", "BasicMotions"])
@pytest.mark.parametrize("return_type", ["numpy3d"])
def test_write_panel_to_tsfile_equal_length(dataset_name, return_type):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = _load_provided_dataset(dataset_name, split="TRAIN", return_type=return_type)
    write_collection_to_tsfile(
        data=X, path="./Temp", target=y, problem_name=dataset_name
    )
    load_path = f"./Temp/{dataset_name}/{dataset_name}.ts"
    newX, newy = load_from_tsfile(
        full_file_path_and_name=load_path, return_type=return_type
    )
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp")


@pytest.mark.parametrize("dataset_name", ["PLAID", "JapaneseVowels"])
def test_write_panel_to_tsfile_unequal_length(dataset_name):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = _load_provided_dataset(
        dataset_name, split="TRAIN", return_type="nested_univ"
    )
    write_collection_to_tsfile(
        data=X, path=f"./Temp{dataset_name}/", target=y, problem_name=dataset_name
    )
    load_path = f"./Temp{dataset_name}/{dataset_name}/{dataset_name}.ts"
    newX, newy = load_from_tsfile(
        full_file_path_and_name=load_path, return_type="nested_univ"
    )
    assert np.array_equal(y, newy)
    shutil.rmtree(f"./Temp{dataset_name}")
