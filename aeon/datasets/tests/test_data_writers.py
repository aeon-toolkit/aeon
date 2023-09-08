# -*- coding: utf-8 -*-
import io
import os
import shutil

import numpy as np
import pandas as pd
import pytest

"""Test functions for data writing."""
from aeon.datasets import load_from_tsfile, write_to_tsfile
from aeon.datasets._data_writers import (
    _write_data_to_tsfile,
    _write_dataframe_to_tsfile,
    _write_header,
    write_results_to_uea_format,
)
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
    path = os.path.dirname(__file__) + "/Temp/"

    write_to_tsfile(
        X=X, path=path, y=y, problem_name=problem_name, regression=regression
    )
    load_path = path + problem_name
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert isinstance(newX, np.ndarray)
    assert X.shape == newX.shape
    assert X[0][0][0] == newX[0][0][0]
    if not regression:
        y = y.astype(str)
        np.testing.assert_array_equal(y, newy)
    else:
        np.testing.assert_array_almost_equal(y, newy)
    shutil.rmtree(path)


def test_fails():
    """Test simple failes return sensible errors and messages."""
    with pytest.raises(TypeError, match="Wrong input data type"):
        write_to_tsfile("FOOBAR", "dummy")
    with pytest.raises(TypeError, match="Data provided must be a ndarray or a list"):
        _write_data_to_tsfile(X="FOOBAR", path="dummy", problem_name="test")
    X = np.random.random(size=(10, 1, 10))
    y = np.array([1, 1, 1, 2, 2, 2])
    with pytest.raises(
        IndexError, match="The number of cases is not the same as the number of labels"
    ):
        _write_data_to_tsfile(X=X, path="dummy", problem_name="test", y=y)
    y_pred = np.array([1, 1, 2, 2])
    with pytest.raises(
        IndexError, match="The number of predicted values is not the same"
    ):
        write_results_to_uea_format(
            "FOO", "BAR", y_pred=y_pred, output_path="dummy", y_true=y
        )
    with pytest.raises(ValueError, match="Cannot have class_labels and targetlabel"):
        _write_header("FOO", "BAR", class_labels=[0, 1], regression=True)
    with pytest.raises(ValueError, match="Data provided must be a DataFrame"):
        _write_dataframe_to_tsfile("FOO", "BAR")


def test__write_header():
    """Test writing an equal length classification and regegression dataset."""
    path = os.path.dirname(__file__) + "/Temp/"
    suffix = "_TRAIN"
    extension = ".ts"
    f = _write_header(
        path=path,
        problem_name="testy",
        suffix=suffix,
        extension=extension,
        series_length=10,
        equal_length=True,
        comment="comment",
        class_labels=[0, 1],
    )
    assert isinstance(f, io.TextIOWrapper)
    f.close()
    file = open(path + "testy" + suffix + extension, "r")
    contents = file.read()
    file.close()
    contents = contents.lower()
    assert "@data" in contents
    assert "@problemname testy" in contents
    assert "@timestamps false" in contents
    assert "@univariate true" in contents
    assert "@equallength true" in contents
    f2 = _write_header(
        path=path,
        problem_name="testy",
        suffix=suffix,
        extension="",
        series_length=10,
        equal_length=True,
        comment="comment",
        regression=True,
    )
    assert isinstance(f2, io.TextIOWrapper)
    f2.close()
    shutil.rmtree(path)


def test__write_dataframe_to_tsfile():
    """Test write dataframe to file.

    Note does not write the header correctly. See
    https://github.com/aeon-toolkit/aeon/issues/732
    """
    path = os.path.dirname(__file__) + "/Temp/"

    X, y = make_nested_dataframe_data(n_cases=10, n_channels=1, n_timepoints=10)
    _write_dataframe_to_tsfile(X, path, problem_name="testy.ts", y=y)
    assert os.path.exists(path + "testy.ts")
    X, y = load_from_tsfile(path + "testy.ts")
    shutil.rmtree(path)


@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_regression_to_tsfile_equal_length(problem_name):
    """Test function to write a regression dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_3d_test_data(regression_target=True)
    path = os.path.dirname(__file__) + "/Temp/"
    write_to_tsfile(X=X, path=path, y=y, problem_name=problem_name)
    load_path = path + problem_name
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert isinstance(newX, np.ndarray)
    assert X.shape == newX.shape
    assert X[0][0][0] == newX[0][0][0]
    y = y.astype(str)
    assert np.array_equal(y, newy)
    shutil.rmtree(path)


@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_to_tsfile_unequal_length(problem_name):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_unequal_length_test_data()
    path = os.path.dirname(__file__) + "/Temp/"
    write_to_tsfile(X=X, path=path, y=y, problem_name=problem_name)
    load_path = path + problem_name
    newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
    assert isinstance(newX, list)
    assert len(X) == len(newX)
    assert X[0][0][0] == newX[0][0][0]
    y = y.astype(str)
    assert np.array_equal(y, newy)
    shutil.rmtree(path)


def test_write_dataframe_to_ts():
    """Tests whether a dataset can be written by the .ts writer then read in."""
    # load an example dataset
    problem_name = "Testy.ts"
    X, y = make_nested_dataframe_data()
    # output the dataframe in a ts file
    path = os.path.dirname(__file__) + "/Temp/"
    _write_dataframe_to_tsfile(
        X=X,
        path=path,
        y=y,
        problem_name=problem_name,
    )
    # load data back from the ts file into dataframe
    newX, newy = load_from_tsfile_to_dataframe(path + problem_name)
    # check if the dataframes are the same
    pd.testing.assert_frame_equal(newX, X)
    y2 = pd.Series(y)
    pd.testing.assert_series_equal(y, y2)
    shutil.rmtree(path)
