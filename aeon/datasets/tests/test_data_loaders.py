"""Test functions for data input and output."""

__maintainer__ = ["TonyBagnall"]

import os
import shutil
import tempfile
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

import aeon
from aeon.datasets import (
    get_dataset_meta_data,
    load_classification,
    load_forecasting,
    load_from_arff_file,
    load_from_ts_file,
    load_from_tsv_file,
    load_regression,
)
from aeon.datasets._data_loaders import (
    CONNECTION_ERRORS,
    _alias_datatype_check,
    _get_channel_strings,
    _load_data,
    _load_header_info,
    _load_saved_dataset,
    _load_tsc_dataset,
    download_dataset,
)
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_forecasting_from_repo():
    """Test load forecasting from repo."""
    name = "FOO"
    with pytest.raises(
        ValueError, match=f"File name {name} is not in the list of " f"valid files"
    ):
        load_forecasting(name)
    name = "m1_quarterly_dataset"
    data, meta = load_forecasting(name, return_metadata=True)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(meta, dict)
    assert meta["frequency"] == "quarterly"
    assert meta["forecast_horizon"] == 8
    assert not meta["contain_missing_values"]
    assert not meta["contain_equal_length"]

    shutil.rmtree(os.path.dirname(__file__) + "/../local_data")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_classification_from_repo():
    """Test load classification from repo."""
    name = "FOO"
    with pytest.raises(ValueError):
        load_classification(name)
    name = "SonyAIBORobotSurface1"
    X, y, meta = load_classification(name, return_metadata=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(meta, dict)
    assert len(X) == len(y)
    assert X.shape == (621, 1, 70)
    assert meta["problemname"] == "sonyaiborobotsurface1"
    assert not meta["timestamps"]
    assert meta["univariate"]
    assert meta["equallength"]
    assert meta["classlabel"]
    assert not meta["targetlabel"]
    assert meta["class_values"] == ["1", "2"]
    shutil.rmtree(os.path.dirname(__file__) + "/../local_data")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_regression_from_repo():
    """Test load regression from repo."""
    name = "FOO"
    with pytest.raises(
        ValueError, match=f"File name {name} is not in the list of " f"valid files"
    ):
        load_regression(name)
    name = "FloodModeling1"
    # name2 = "ParkingBirmingham"
    # name3 = "AcousticContaminationMadrid"
    with tempfile.TemporaryDirectory() as tmp:
        X, y, meta = load_regression(name, extract_path=tmp, return_metadata=True)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(meta, dict)
        assert len(X) == len(y)
        assert X.shape == (673, 1, 266)
        assert meta["problemname"] == "floodmodeling1"
        assert not meta["timestamps"]
        assert meta["univariate"]
        assert meta["equallength"]
        assert not meta["missing"]
        assert not meta["classlabel"]
        assert meta["targetlabel"]
        assert meta["class_values"] == []
        # TODO restore these tests once these data are on zenodo and or tsc.com works.
        # # Test load equal length
        # X, y, meta = load_regression(
        #     name2, extract_path=tmp, return_metadata=True, load_equal_length=True
        # )
        # assert meta["equallength"]
        # X, y, meta = load_regression(
        #     name2, extract_path=tmp, return_metadata=True, load_equal_length=False
        # )
        # assert not meta["equallength"]
        # # Test load no missing values
        # X, y, meta = load_regression(
        #     name3, extract_path=tmp, return_metadata=True, load_no_missing=True
        # )
        # assert not meta["missing"]
        # X, y, meta = load_regression(
        #     name3, extract_path=tmp, return_metadata=True, load_no_missing=False
        # )
        # assert meta["missing"]


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_fails():
    """Test load fails."""
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/",
    )
    with pytest.raises(ValueError):
        load_regression("FOOBAR", extract_path=data_path)
    with pytest.raises(ValueError):
        load_classification("FOOBAR", extract_path=data_path)
    with pytest.raises(ValueError):
        load_forecasting("FOOBAR", extract_path=data_path)


def test__alias_datatype_check():
    """Test the alias check."""
    assert _alias_datatype_check("FOO") == "FOO"
    assert _alias_datatype_check("np2d") == "numpy2D"
    assert _alias_datatype_check("numpy2d") == "numpy2D"
    assert _alias_datatype_check("numpyflat") == "numpy2D"
    assert _alias_datatype_check("numpy3d") == "numpy3D"
    assert _alias_datatype_check("np3d") == "numpy3D"
    assert _alias_datatype_check("np3D") == "numpy3D"


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test__load_header_info():
    """Test load header info."""
    path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTest_TRAIN.ts",
    )
    """Test loading a header."""
    with open(path, encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        assert meta_data["problemname"] == "unittest"
        assert not meta_data["timestamps"]
        assert not meta_data["missing"]
        assert meta_data["univariate"]
        assert meta_data["equallength"]
        assert meta_data["classlabel"]
        assert meta_data["class_values"][0] == "1"
        assert meta_data["class_values"][1] == "2"
    WRONG_STRINGS = ["@data 55", "@missing 42, @classlabel True"]
    count = 1
    with tempfile.TemporaryDirectory() as tmp:
        for name in WRONG_STRINGS:
            problem_name = f"temp{count}.ts"
            load_path = os.path.join(tmp, problem_name)
            temp_file = open(load_path, "w", encoding="utf-8")
            temp_file.write(name)
            temp_file.close()
            with open(load_path, encoding="utf-8") as file:
                with pytest.raises(IOError):
                    _load_header_info(file)
            count = count + 1
        name = "@missing true \n @classlabel True 0 1"
        problem_name = "temp_correct.ts"
        load_path = os.path.join(tmp, problem_name)
        temp_file = open(load_path, "w", encoding="utf-8")
        temp_file.write(name)
        temp_file.close()
        with open(load_path, encoding="utf-8") as file:
            meta = _load_header_info(file)
            assert meta["missing"] is True
            assert meta["classlabel"] is True


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test__load_data():
    """Test loading after header."""
    path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTest_TRAIN.ts",
    )
    with open(path, encoding="utf-8") as file:
        meta_data = _load_header_info(file)
        X, y, _ = _load_data(file, meta_data)
        assert X.shape == (20, 1, 24)
        assert len(y) == 20
    path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/BasicMotions/BasicMotions_TRAIN.ts",
    )
    with open(path, encoding="utf-8") as file:
        meta_data = _load_header_info(file)
        # Check raise error for incorrect univariate test
        meta_data["univariate"] = True
        with pytest.raises(IOError):
            X, y, _ = _load_data(file, meta_data)
    path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/JapaneseVowels/JapaneseVowels_TRAIN.ts",
    )
    with open(path, encoding="utf-8") as file:
        meta_data = _load_header_info(file)
        # Check raise error for incorrect univariate test
        meta_data["equallength"] = True
        with pytest.raises(IOError):
            X, y, _ = _load_data(file, meta_data)
    WRONG_DATA = [
        "1.0,2.0,3.0:1.0,2.0,3.0, 4.0:0",
        "1.0,2.0,3.0:1.0,2.0,3.0, 4.0:0\n1.0,2.0,3.0,4.0:0",
    ]
    meta_data = {
        "classlabel": True,
        "class_values": [0, 1],
        "equallength": False,
        "timestamps": False,
    }
    count = 1
    with tempfile.TemporaryDirectory() as tmp:
        for data in WRONG_DATA:
            problem_name = f"temp{count}.ts"
            load_path = os.path.join(tmp, problem_name)
            temp_file = open(load_path, "w", encoding="utf-8")
            temp_file.write(data)
            temp_file.close()
            with open(load_path, encoding="utf-8") as file:
                with pytest.raises(IOError):
                    _load_data(file, meta_data)
            count = count + 1
        meta_data = {
            "classlabel": True,
            "class_values": [0, 1],
            "equallength": True,
            "univariate": True,
        }
        data = "1.0,2.0,3.0:0.0\n 1.0,2.0,3.0:1.0\n 2.0,3.0,4.0:0"
        problem_name = "tempCorrect.ts"
        load_path = os.path.join(tmp, problem_name)
        temp_file = open(load_path, "w", encoding="utf-8")
        temp_file.write(data)
        temp_file.close()
        with open(load_path, encoding="utf-8") as file:
            X, y, meta_data = _load_data(file, meta_data)
            assert isinstance(X, np.ndarray)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("return_type", ["numpy3D", "numpy2D"])
def test_load_provided_dataset(return_type):
    """Test function to check for proper loading.

    Check all possibilities of  return_type.
    """
    X, y = _load_saved_dataset("UnitTest", "TRAIN", return_type)
    if return_type == "numpy3D":
        assert isinstance(X, np.ndarray) and X.ndim == 3
    elif return_type == "numpy2D":
        assert isinstance(X, np.ndarray) and X.ndim == 2


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_ts_file():
    """Test function for loading TS formats.

    Test
    1. Univariate equal length (UnitTest) returns 3D numpy X, 1D numpy y
    2. Multivariate equal length (BasicMotions) returns 3D numpy X, 1D numpy y
    3. Univariate and multivariate unequal length (PLAID) return X as list of numpy
    """
    # Test 1.1: load univariate equal length (UnitTest), should return 2D array and 1D
    # array, test first and last data
    # Test 1.2: Load a problem without y values (UnitTest),  test first and last data.
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTest_TRAIN.ts",
    )
    X, y = load_from_ts_file(data_path, return_meta_data=False)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert X.shape == (20, 1, 24) and y.shape == (20,)
    assert X[0][0][0] == 573.0
    X, y = load_from_ts_file(data_path, return_meta_data=False, return_type="numpy2D")
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.shape == (20, 24)
    assert X[0][0] == 573.0

    # Test 2: load multivare equal length (BasicMotions), should return 3D array and 1D
    # array, test first and last data.
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/BasicMotions/BasicMotions_TRAIN.ts",
    )
    X, y = load_from_ts_file(data_path, return_meta_data=False)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (40, 6, 100) and y.shape == (40,)
    assert X[1][2][3] == -1.898794
    X, y = load_from_ts_file(data_path, return_meta_data=False)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert X.shape == (40, 6, 100) and y.shape == (40,)
    assert X[1][2][3] == -1.898794

    # Test 3.1: load univariate unequal length (PLAID), should return a one column
    # dataframe,
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/PLAID/PLAID_TRAIN.ts",
    )

    X, y = load_from_ts_file(full_file_path_and_name=data_path, return_meta_data=False)
    assert isinstance(X, list) and isinstance(y, np.ndarray)
    assert len(X) == 537 and y.shape == (537,)
    # Test 3.2: load multivariate unequal length (JapaneseVowels), should return a X
    # columns dataframe,
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/JapaneseVowels/JapaneseVowels_TRAIN.ts",
    )
    X, y = load_from_ts_file(full_file_path_and_name=data_path, return_meta_data=False)
    assert isinstance(X, list) and isinstance(y, np.ndarray)
    assert len(X) == 270 and y.shape == (270,)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_regression():
    """Test the load regression function."""
    expected_metadata = {
        "problemname": "covid3month",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equallength": True,
        "targetlabel": True,
        "classlabel": False,
        "class_values": [],
    }
    X, y, meta = load_regression("Covid3Month", return_metadata=True)
    assert meta == expected_metadata
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (201, 1, 84)
    assert y.shape == (201,)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_classification():
    """Test load classification."""
    expected_metadata = {
        "problemname": "unittest",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equallength": True,
        "targetlabel": False,
        "classlabel": True,
        "class_values": ["1", "2"],
    }
    X, y, meta = load_classification("UnitTest", return_metadata=True)
    assert meta == expected_metadata
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (42, 1, 24)
    assert y.shape == (42,)
    # Try load covid, should work
    X, y, meta = load_classification("Covid3Month", return_metadata=True)

    with pytest.raises(ValueError, match="You have tried to load a regression problem"):
        X, y = load_classification("CardanoSentiment")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_ucr_tsv():
    """Test that GunPoint is the same when loaded from .ts and .tsv."""
    X, y = _load_saved_dataset("GunPoint", split="TRAIN")
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/GunPoint/GunPoint_TRAIN.tsv",
    )
    X2, y2 = load_from_tsv_file(data_path)
    y = y.astype(float)
    np.testing.assert_array_almost_equal(X, X2, decimal=4)
    assert np.array_equal(y, y2)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_load_from_arff():
    """Test that GunPoint is the same when loaded from .ts and .arff."""
    X, y = _load_saved_dataset("GunPoint", split="TRAIN")
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/GunPoint/GunPoint_TRAIN.arff",
    )
    X2, y2 = load_from_arff_file(data_path)
    assert isinstance(X2, np.ndarray)
    assert isinstance(y2, np.ndarray)
    assert X.shape == X2.shape
    assert len(X2) == len(y2)
    np.testing.assert_array_almost_equal(X, X2, decimal=4)
    assert np.array_equal(y, y2)
    X, y = _load_saved_dataset("BasicMotions", split="TRAIN")
    data_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/BasicMotions/BasicMotions_TRAIN.arff",
    )
    X2, y2 = load_from_arff_file(data_path)
    assert isinstance(X, np.ndarray)
    assert isinstance(y2, np.ndarray)
    np.testing.assert_array_almost_equal(X, X2, decimal=4)


def test__get_channel_strings():
    """Test get channel string."""
    line = "(2007-01-01 00:00:00,241.97),(2007-01-01 00:01:00,241.75):1"
    channel_strings = _get_channel_strings(line)
    assert len(channel_strings) == 2
    assert channel_strings[0] == "241.97,241.75"


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=(URLError, TimeoutError, ConnectionError))
def test_get_meta_data():
    """Test the get_dataset_meta_data function."""
    df = get_dataset_meta_data()
    assert isinstance(df, pd.DataFrame)
    df = get_dataset_meta_data(features="TrainSize")
    assert df.shape[1] == 2
    df = get_dataset_meta_data(data_names=["Adiac", "Chinatown"])
    assert df.shape[0] == 2
    df = get_dataset_meta_data(
        data_names=["Adiac", "Chinatown"], features=["TestSize", "Channels"]
    )
    assert df.shape == (2, 3)
    with pytest.raises(ValueError):
        df = get_dataset_meta_data(url="FOOBAR")


def test__load_saved_dataset():
    """Test parameter settings for loading a saved dataset."""
    name = "UnitTest"
    local_dirname = "data"
    X, y = _load_saved_dataset(name)
    X2, y = _load_saved_dataset(name, local_dirname=local_dirname)
    X3, y = _load_saved_dataset(name, dir_name="UnitTest", local_dirname=local_dirname)
    X4, y = _load_saved_dataset(
        name, dir_name="UnitTest", split="Train", local_dirname=local_dirname
    )
    X5, y = _load_saved_dataset(name, dir_name="UnitTest", split="Train")

    assert np.array_equal(X, X2)
    assert np.array_equal(X, X3)
    assert np.array_equal(X4, X5)
    assert not np.array_equal(X, X4)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=(URLError, TimeoutError, ConnectionError))
def test_download_dataset():
    """Test the private download_dataset function."""
    name = "Chinatown"
    with tempfile.TemporaryDirectory() as tmp:
        path = download_dataset(name, save_path=tmp)
        assert path == os.path.join(tmp, name)
        path = download_dataset(name, save_path=tmp)
        assert path == os.path.join(tmp, name)
        with pytest.raises(ValueError, match="Invalid dataset name"):
            download_dataset("FOO", save_path=tmp)
        with pytest.raises(ValueError, match="Invalid dataset name"):
            download_dataset("BAR")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=(URLError, TimeoutError, ConnectionError))
def test_load_tsc_dataset():
    """Test the private _load_tsc_dataset function."""
    name = "Chinatown"
    with tempfile.TemporaryDirectory() as tmp:
        X, y = _load_tsc_dataset(name, split="TRAIN", extract_path=tmp)
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        with pytest.raises(ValueError, match="Invalid dataset name"):
            _load_tsc_dataset("FOO", split="TEST", extract_path=tmp)
