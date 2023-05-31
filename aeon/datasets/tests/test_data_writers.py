# -*- coding: utf-8 -*-
import shutil

import numpy as np

"""Test functions for data writing."""
from aeon.datasets import load_from_tsfile, write_to_tsfile
from aeon.datasets._data_loaders import load_from_tsfile_to_dataframe
from aeon.datasets._data_writers import _write_dataframe_to_tsfile
from aeon.utils._testing.collection import (
    make_3d_test_data,
    make_nested_dataframe_data,
    make_unequal_length_test_data,
)


def test_write_to_tsfile_equal_length():
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_3d_test_data()
    write_to_tsfile(X=X, path="./Temp/", y=y, problem_name="Testy")
    load_path = "./Temp/Testy.ts"
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert X.shape == newX.shape
    assert X[0][0][0] == newX[0][0][0]
    y = y.astype(str)
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp")


def test_write_to_tsfile_unequal_length():
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_unequal_length_test_data()
    write_to_tsfile(X=X, path="./Temp/", y=y, problem_name="Testy2")
    load_path = "./Temp/Testy2.ts"
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert len(X) == len(newX)
    assert X[0][0][0] == newX[0][0][0]
    y = y.astype(str)
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp")


def test_write_dataframe_to_ts():
    """Tests whether a dataset can be written by the .ts writer then read in."""
    # load an example dataset
    X, y = make_nested_dataframe_data()
    # output the dataframe in a ts file
    dataset_name = "Testy.ts"
    _write_dataframe_to_tsfile(
        X=X,
        path="./Temp/",
        y=y,
        problem_name=dataset_name,
        equal_length=True,
    )
    # load data back from the ts file into dataframe
    newX, newy = load_from_tsfile_to_dataframe("./Temp/")
    # check if the dataframes are the same
    #    assert_frame_equal(newX, X)
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp/")
