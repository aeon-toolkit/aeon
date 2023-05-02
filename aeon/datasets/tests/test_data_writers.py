# -*- coding: utf-8 -*-
import shutil

import numpy as np
import pytest

"""Test functions for data writing."""
from aeon.datasets import load_from_tsfile, write_to_tsfile
from aeon.datasets._data_loaders import _load_provided_dataset
from aeon.datasets._data_writers import _write_dataframe_to_tsfile


@pytest.mark.parametrize("dataset_name", ["UnitTest", "BasicMotions"])
def test_write_to_tsfile_equal_length(dataset_name):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = _load_provided_dataset(dataset_name, split="TRAIN")
    write_to_tsfile(X=X, path="./Temp", y=y, problem_name=dataset_name)
    load_path = f"./Temp/{dataset_name}/{dataset_name}.ts"
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert X.shape == newX.shape
    assert X[0][0][0] == newX[0][0][0]
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp")


@pytest.mark.parametrize("dataset_name", ["PLAID", "JapaneseVowels"])
def test_write_to_tsfile_unequal_length(dataset_name):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = _load_provided_dataset(dataset_name, split="TRAIN")
    write_to_tsfile(X=X, path=f"./Temp{dataset_name}/", y=y, problem_name=dataset_name)
    load_path = f"./Temp{dataset_name}/{dataset_name}/{dataset_name}.ts"
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert len(X) == len(newX)
    assert X[0][0][0] == newX[0][0][0]
    assert np.array_equal(y, newy)
    shutil.rmtree(f"./Temp{dataset_name}")


@pytest.mark.parametrize("dataset_name", ["UnitTest", "BasicMotions"])
def test_write_dataframe_to_ts(dataset_name):
    """Tests whether a dataset can be written by the .ts writer then read in."""
    # load an example dataset
    X, y = _load_provided_dataset(
        dataset_name, split="TRAIN", return_type="nested_univ"
    )
    # output the dataframe in a ts file
    _write_dataframe_to_tsfile(
        X=X,
        path=f"./Temp{dataset_name}/",
        y=y,
        problem_name=dataset_name,
        equal_length=True,
        suffix="_TRAIN",
    )
    # load data back from the ts file


#    load_path = f"./Temp{dataset_name}/{dataset_name}/{dataset_name}_TRAIN.ts"


#    newX, newy = _load_from_tsfile_to_dataframe(load_path)
# check if the dataframes are the same
#    assert_frame_equal(newX, X)
#    assert np.array_equal(y, newy)
