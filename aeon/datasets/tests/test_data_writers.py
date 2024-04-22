"""Test functions for data writing."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from aeon.datasets import load_from_arff_file, load_from_tsfile, write_to_tsfile
from aeon.datasets._data_writers import (
    _write_data_to_tsfile,
    _write_dataframe_to_tsfile,
    _write_header,
    write_results_to_uea_format,
    write_to_arff_file,
)
from aeon.datasets._dataframe_loaders import load_from_tsfile_to_dataframe
from aeon.testing.test_config import PR_TESTING
from aeon.testing.utils.data_gen import (
    make_example_3d_numpy,
    make_example_nested_dataframe,
    make_example_unequal_length,
)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("regression", [True, False])
@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_to_tsfile_equal_length(regression, problem_name):
    """Test function to write an equal length classification and regegression dataset.

    creates an equal length problem, writes locally, reloads, then compares data. It
    then deletes the files.
    """
    X, y = make_example_3d_numpy(regression_target=regression)
    with tempfile.TemporaryDirectory() as tmp:
        write_to_tsfile(
            X=X, path=tmp, y=y, problem_name=problem_name, regression=regression
        )
        load_path = os.path.join(tmp, problem_name)
        newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
        assert isinstance(newX, np.ndarray)
        assert X.shape == newX.shape
        assert X[0][0][0] == newX[0][0][0]
        if not regression:
            y = y.astype(str)
            np.testing.assert_array_equal(y, newy)
        else:
            np.testing.assert_array_almost_equal(y, newy)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_regression_to_tsfile_equal_length(problem_name):
    """Test function to write a regression dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_example_3d_numpy(regression_target=True)
    with tempfile.TemporaryDirectory() as tmp:
        write_to_tsfile(X=X, path=tmp, y=y, problem_name=problem_name)
        load_path = os.path.join(tmp, problem_name)
        newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
        assert isinstance(newX, np.ndarray)
        assert X.shape == newX.shape
        assert X[0][0][0] == newX[0][0][0]
        y = y.astype(str)
        assert np.array_equal(y, newy)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_to_tsfile_unequal_length(problem_name):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_example_unequal_length()
    with tempfile.TemporaryDirectory() as tmp:
        write_to_tsfile(X=X, path=tmp, y=y, problem_name=problem_name)
        load_path = os.path.join(tmp, problem_name)
        newX, newy = load_from_tsfile(full_file_path_and_name=load_path)
        assert isinstance(newX, list)
        assert len(X) == len(newX)
        assert X[0][0][0] == newX[0][0][0]
        y = y.astype(str)
        assert np.array_equal(y, newy)


def test_write_data_to_tsfile_invalid():
    """Test function to check the handling of invalid inputs by write_to_tsfile."""
    with pytest.raises(TypeError, match="Wrong input data type"):
        write_to_tsfile("A string", "path")
    with pytest.raises(TypeError, match="Data provided must be a ndarray or a list"):
        _write_data_to_tsfile("AFC", "49", "undefeated")
    X, _ = make_example_3d_numpy(n_cases=6, n_timepoints=10, n_channels=1)
    y = np.ndarray([0, 1, 1, 0, 1])
    with pytest.raises(
        IndexError,
        match="The number of cases in X does not match the number of values in y",
    ):
        _write_data_to_tsfile(X, "temp", "temp", y=y)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("tsfile_writer", [_write_dataframe_to_tsfile, write_to_tsfile])
def test_write_dataframe_to_ts(tsfile_writer):
    """Tests whether a dataset can be written by the .ts writer then read in."""
    # load an example dataset
    problem_name = "Testy.ts"
    X, y = make_example_nested_dataframe()
    with tempfile.TemporaryDirectory() as tmp:
        # output the dataframe in a ts file
        tsfile_writer(
            X=X,
            path=tmp,
            y=y,
            problem_name=problem_name,
        )
        # load data back from the ts file into dataframe
        load_path = os.path.join(tmp, problem_name)
        newX, newy = load_from_tsfile_to_dataframe(load_path)
        # check if the dataframes are the same
        pd.testing.assert_frame_equal(newX, X)
        y2 = pd.Series(y)
        pd.testing.assert_series_equal(y, y2)


def test_write_header():
    """Test _write_header."""
    with tempfile.TemporaryDirectory() as tmp:
        problem_name = "header.csv"
        with pytest.raises(
            ValueError, match="Cannot have class_labels true for a regression problem"
        ):
            _write_header(tmp, problem_name, class_labels=True, regression=True)
    _write_header(
        tmp,
        problem_name,
        suffix="_TRAIN",
        extension=".csv",
        comment="Hello",
        regression=True,
    )


def test_write_to_arff_file():
    """Test function to check writing into an ARFF file and loading from it."""
    X, y = make_example_3d_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        write_to_arff_file(X, y, tmp, problem_name="Test_arff", header="Description")

        load_path = os.path.join(tmp, "Test_arff.arff")
        X_new, y_new = load_from_arff_file(full_file_path_and_name=load_path)

        assert isinstance(X_new, np.ndarray)
        assert X.shape == X_new.shape
        assert_array_equal(X, X_new)
        assert_array_equal(y.astype(str), y_new)


def test_write_results_to_uea_format():
    """Test function to check writing results into UEA format."""
    with tempfile.TemporaryDirectory() as tmp:
        y_true = np.array([0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0])
        with pytest.raises(
            IndexError, match="The number of predicted values is not the same"
        ):
            write_results_to_uea_format(
                "HC", "Testy", y_pred=y_pred, y_true=y_true, output_path=tmp
            )
        y_true = np.array([0, 1, 1, 0])
        write_results_to_uea_format(
            "HC",
            "Testy",
            y_pred=y_pred,
            output_path=tmp,
            full_path=False,
            split="TEST",
            timing_type="seconds",
            first_line_comment="Hello",
        )

        probs = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        write_results_to_uea_format(
            "HC",
            "Testy2",
            y_pred=y_pred,
            y_true=y_true,
            output_path=tmp,
            full_path=False,
            split="TEST",
            timing_type="seconds",
            first_line_comment="Hello",
            predicted_probs=probs,
        )
