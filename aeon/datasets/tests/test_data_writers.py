# -*- coding: utf-8 -*-
import shutil

import numpy as np
import pandas as pd
import pytest

"""Test functions for data writing."""
from aeon.datasets import load_from_tsfile, write_to_tsfile
from aeon.datasets._data_writers import _write_dataframe_to_tsfile
from aeon.datasets._dataframe_loaders import load_from_tsfile_to_dataframe
from aeon.utils._testing.collection import (
    make_3d_test_data,
    make_nested_dataframe_data,
    make_unequal_length_test_data,
)


@pytest.mark.parametrize("regression", [True, False])
@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_to_tsfile_equal_length(regression, problem_name):
    """Test function to write an equal length classification and regegression dataset.

    creates an equal length problem, writes locally, reloads, then compares data. It
    then deletes the files.
    """
    X, y = make_3d_test_data(regression_target=regression)
    write_to_tsfile(
        X=X, path="./Temp/", y=y, problem_name=problem_name, regression=regression
    )
    load_path = "./Temp/" + problem_name
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert isinstance(newX, np.ndarray)
    assert X.shape == newX.shape
    assert X[0][0][0] == newX[0][0][0]
    if not regression:
        y = y.astype(str)
        np.testing.assert_array_equal(y, newy)
    else:
        np.testing.assert_array_almost_equal(y, newy)
    shutil.rmtree("./Temp")


@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_regression_to_tsfile_equal_length(problem_name):
    """Test function to write a regression dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_3d_test_data(regression_target=True)
    write_to_tsfile(X=X, path="./Temp/", y=y, problem_name=problem_name)
    load_path = "./Temp/" + problem_name
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert isinstance(newX, np.ndarray)
    assert X.shape == newX.shape
    assert X[0][0][0] == newX[0][0][0]
    y = y.astype(str)
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp")


@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_to_tsfile_unequal_length(problem_name):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_unequal_length_test_data()
    write_to_tsfile(X=X, path="./Temp/", y=y, problem_name=problem_name)
    load_path = "./Temp/" + problem_name
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert isinstance(newX, list)
    assert len(X) == len(newX)
    assert X[0][0][0] == newX[0][0][0]
    y = y.astype(str)
    assert np.array_equal(y, newy)
    shutil.rmtree("./Temp")


def test_write_dataframe_to_ts():
    """Tests whether a dataset can be written by the .ts writer then read in."""
    # load an example dataset
    problem_name = "Testy.ts"
    X, y = make_nested_dataframe_data()
    # output the dataframe in a ts file
    _write_dataframe_to_tsfile(
        X=X,
        path="./Temp/",
        y=y,
        problem_name=problem_name,
    )
    # load data back from the ts file into dataframe
    newX, newy = load_from_tsfile_to_dataframe("./Temp/" + problem_name)
    # check if the dataframes are the same
    pd.testing.assert_frame_equal(newX, X)
    y2 = pd.Series(y)
    pd.testing.assert_series_equal(y, y2)
    shutil.rmtree("./Temp/")
