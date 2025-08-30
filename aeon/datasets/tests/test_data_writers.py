"""Test functions for data writing."""

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon.datasets import (
    load_from_arff_file,
    load_from_ts_file,
    save_to_ts_file,
    write_to_arff_file,
)
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.testing_config import PR_TESTING
from aeon.testing.testing_data import MISSING_VALUES_CLASSIFICATION
from aeon.utils.validation.collection import has_missing, is_univariate


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("regression", [True, False])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_save_to_ts_file_equal_length(regression, n_channels):
    """Test function to write an equal length classification and regression dataset.

    Creates an equal length problem, writes locally, reloads, then compares data. It
    then deletes the files.
    """
    X, y = make_example_3d_numpy(regression_target=regression, n_channels=n_channels)
    with tempfile.TemporaryDirectory() as tmp:
        save_to_ts_file(
            X=X,
            y=y,
            path=tmp,
            problem_name=f"test_{regression}",
            label_type="regression" if regression else "classification",
        )
        load_path = os.path.join(tmp, f"test_{regression}.ts")
        new_X, new_y = load_from_ts_file(full_file_path_and_name=load_path)
        assert isinstance(new_X, np.ndarray)
        assert isinstance(new_y, np.ndarray)
        assert is_univariate(X) == (n_channels == 1)
        assert is_univariate(new_X) == (n_channels == 1)
        np.testing.assert_array_almost_equal(X, new_X)
        np.testing.assert_array_almost_equal(y, new_y.astype(float))


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("n_channels", [1, 3])
def test_save_to_ts_file_unequal_length(n_channels):
    """Test function to write an unequal length dataset.

    Creates an unequal length problem, writes locally, reloads, then compares data. It
    then deletes the files.
    """
    X, y = make_example_3d_numpy_list(n_channels=n_channels)
    with tempfile.TemporaryDirectory() as tmp:
        save_to_ts_file(
            X=X, y=y, path=tmp, problem_name="test", label_type="classification"
        )
        load_path = os.path.join(tmp, "test.ts")
        new_X, new_y = load_from_ts_file(full_file_path_and_name=load_path)
        assert isinstance(new_X, list)
        assert len(X) == len(new_X)
        assert isinstance(new_y, np.ndarray)
        assert is_univariate(X) == (n_channels == 1)
        assert is_univariate(new_X) == (n_channels == 1)
        for i in range(len(new_X)):
            assert isinstance(new_X[i], np.ndarray)
            np.testing.assert_array_almost_equal(X[i], new_X[i])
        np.testing.assert_array_equal(y, new_y.astype(float))


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_save_to_ts_file_missing_values():
    """Test function to write an equal length classification dataset with NaNs.

    Writes test data locally, reloads, then compares data. It then deletes the files.
    """
    X, y = MISSING_VALUES_CLASSIFICATION["numpy3D"]["train"]
    with tempfile.TemporaryDirectory() as tmp:
        save_to_ts_file(
            X=X,
            y=y,
            path=tmp,
            problem_name="test_missing",
            label_type="classification",
        )
        load_path = os.path.join(tmp, "test_missing.ts")
        new_X, new_y = load_from_ts_file(full_file_path_and_name=load_path)
        assert isinstance(new_X, np.ndarray)
        assert isinstance(new_y, np.ndarray)
        assert has_missing(X)
        assert has_missing(new_X)
        np.testing.assert_array_almost_equal(X, new_X)
        np.testing.assert_array_almost_equal(y, new_y.astype(float))


def test_save_data_to_ts_file_invalid():
    """Test function to check the handling of invalid inputs by save_to_ts_file."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(TypeError, match="Wrong input data type"):
            save_to_ts_file("A string", path=tmp)

        X, _ = make_example_3d_numpy(n_cases=6, n_timepoints=10, n_channels=1)
        y = np.array([0, 1, 1, 0, 1])
        with pytest.raises(
            ValueError,
            match="If y is not None, label_type must be either",
        ):
            save_to_ts_file(X, y, path=tmp)
        with pytest.raises(
            ValueError,
            match="The number of cases in X does not match the number of values in y",
        ):
            save_to_ts_file(X, y, path=tmp, label_type="classification")


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


def test_write_to_arff_wrong_inputs():
    """Tests whether error thrown if wrong input."""
    # load an example dataset
    with tempfile.TemporaryDirectory() as tmp:
        X = "A string"
        y = [1, 2, 3, 4]
        with pytest.raises(TypeError, match="Wrong input data type"):
            write_to_arff_file(X, y, tmp)
        X2, y2 = make_example_3d_numpy(n_cases=5, n_channels=2)
        with pytest.raises(ValueError, match="must be a 3D array with shape"):
            write_to_arff_file(X2, y2, tmp)
