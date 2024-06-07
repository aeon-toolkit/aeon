"""Test anomaly detection dataset loaders."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from aeon.datasets._tsad_data_loaders import (
    _DATA_FOLDER,
    load_anomaly_detection,
    load_Daphnet_S07R01E0,
    load_ecg_diff_count_3,
    load_from_timeeval_csv_file,
    load_kdd_tsad_135,
)


def test_load_anomaly_detection_wrong_name():
    """Test load non-existent anomaly detection datasets."""
    with pytest.raises(
        ValueError, match="The name of the dataset must be a tuple of two strings.*"
    ):
        load_anomaly_detection(("FOO", "BAR", "BAZ"))
    with pytest.raises(
        ValueError,
        match="When loading a custom dataset, the extract_path must point "
        "to a TimeEval-formatted CSV file.*",
    ):
        load_anomaly_detection(("FOO", "BAR"))


def test_load_anomaly_detection_no_train_split():
    """Test load train split of anomaly detection dataset without training split."""
    name = ("Dodgers", "101-freeway-traffic")
    with pytest.raises(
        ValueError, match="Dataset .* does not have a training partition.*"
    ):
        load_anomaly_detection(name, split="train")


# def test_load_anomaly_detection_from_repo():
#     """Test load univariate anomaly detection dataset from repo."""
#     name = ("Dodgers", "101-freeway-traffic")
#     X, y, meta = load_anomaly_detection(name, return_metadata=True)
#     assert isinstance(X, np.ndarray)
#     assert X.shape == (50400,)
#     assert isinstance(y, np.ndarray)
#     assert y.shape == (50400,)
#     assert isinstance(meta, dict)
#     assert meta["learning_type"] == "unsupervised"
#     assert meta["num_anomalies"] == 133
#     np.testing.assert_almost_equal(meta["contamination"], 0.1113, decimal=4)
#
#
# def test_load_anomaly_detection_from_repo_multivariate():
#     """Test load multivariate anomaly detection dataset from repo."""
#     name = ("CalIt2", "CalIt2-traffic")
#     X, y, meta = load_anomaly_detection(name, return_metadata=True)
#     assert isinstance(X, np.ndarray)
#     assert X.shape == (5040, 2)
#     assert isinstance(y, np.ndarray)
#     assert y.shape == (5040,)
#     assert isinstance(meta, dict)
#     assert meta["learning_type"] == "unsupervised"
#     assert meta["num_anomalies"] == 29
#     np.testing.assert_almost_equal(meta["contamination"], 0.0409, decimal=4)


def test_load_anomaly_detection_from_archive():
    """Test load anomaly detection dataset from archive."""
    name = ("Genesis", "genesis-anomalies")
    X, y, meta = load_anomaly_detection(name, return_metadata=True)
    assert isinstance(X, np.ndarray)
    assert X.shape == (16220, 18)
    assert isinstance(y, np.ndarray)
    assert y.shape == (16220,)
    assert isinstance(meta, dict)
    assert meta["learning_type"] == "unsupervised"
    assert meta["num_anomalies"] == 3
    np.testing.assert_almost_equal(meta["contamination"], 0.0031, decimal=4)

    shutil.rmtree(_DATA_FOLDER / "multivariate" / "Genesis")


def test_load_anomaly_detection_unavailable():
    """Test load anomaly detection dataset that is not available."""
    name = ("IOPS", "05f10d3a-239c-3bef-9bdc-a2feeb0037aa")
    with pytest.raises(
        ValueError, match=f"Collection {name[0]} .* not available for download.*"
    ):
        load_anomaly_detection(name)


def test_load_anomaly_detection_custom_file():
    """Test load anomaly detection dataset from custom file."""
    name = ("correlation-anomalies", "rmj-2-short-2-diff-channel")
    with pytest.raises(
        ValueError,
        match="When loading a custom dataset, the extract_path must point to a "
        "TimeEval-formatted CSV file.*",
    ):
        load_anomaly_detection(name)

    def _test(filename: str, learning_type: str) -> None:
        X, y, meta = load_anomaly_detection(
            name,
            extract_path=Path(__file__).parent.parent / "data" / "UnitTest" / filename,
            split="test" if learning_type == "unsupervised" else "train",
            return_metadata=True,
        )
        assert X.shape[0] == y.shape[0]
        assert meta["learning_type"] == learning_type
        assert meta["num_anomalies"] == 0 if learning_type == "semi-supervised" else 3

    _test("ecg-diff-count-3_TEST.csv", "unsupervised")
    _test("ecg-diff-count-3_TRAIN_NA.csv", "semi-supervised")
    _test("ecg-diff-count-3_TRAIN_A.csv", "supervised")


def test_load_from_timeeval_csv_file_univariate():
    """Test load univariate dataset from file."""
    X, y = load_from_timeeval_csv_file(
        Path(__file__).parent.parent
        / "data"
        / "KDD-TSAD_135"
        / "135_UCR_Anomaly_InternalBleeding16_TEST.csv"
    )
    assert isinstance(X, np.ndarray)
    assert X.shape == (7501,)
    assert isinstance(y, np.ndarray)
    assert y.shape == (7501,)


def test_load_from_timeeval_csv_file_multivariate():
    """Test load multivariate dataset from file."""
    X, y = load_from_timeeval_csv_file(
        Path(__file__).parent.parent / "data" / "Daphnet_S07R02E0" / "S07R02E0.csv"
    )
    assert isinstance(X, np.ndarray)
    assert X.shape == (28800, 9)
    assert isinstance(y, np.ndarray)
    assert y.shape == (28800,)


def test_load_kdd_tsad_135():
    """Test load KDD TSAD 135 dataset."""
    X, y = load_kdd_tsad_135()
    assert isinstance(X, np.ndarray)
    assert X.shape == (7501,)
    assert isinstance(y, np.ndarray)
    assert y.shape == (7501,)

    X_train, y_train = load_kdd_tsad_135(split="train")
    assert isinstance(X_train, np.ndarray)
    assert X_train.shape == (1200,)
    assert isinstance(y_train, np.ndarray)
    assert y_train.shape == (1200,)


def test_load_Daphnet_S07R01E0():
    """Test load Daphnet S07R01E0 dataset."""
    X, y = load_Daphnet_S07R01E0()
    assert isinstance(X, np.ndarray)
    assert X.shape == (28800, 9)
    assert isinstance(y, np.ndarray)
    assert y.shape == (28800,)


@pytest.mark.parametrize(
    "learning_type", ["unsupervised", "semi-supervised", "supervised"]
)
def test_load_ecg_diff_count_3(learning_type):
    """Test load ecg-diff-count-3 dataset."""
    if learning_type == "unsupervised":
        X, y = load_ecg_diff_count_3(learning_type)
    else:
        X, y, X_train, y_train = load_ecg_diff_count_3(learning_type)
        assert isinstance(X_train, np.ndarray)
        assert X_train.shape == (10000,)
        assert isinstance(y_train, np.ndarray)
        assert y_train.shape == (10000,)
        if learning_type == "semi-supervised":
            assert np.sum(y_train) == 0

    assert isinstance(X, np.ndarray)
    assert X.shape == (10000,)
    assert isinstance(y, np.ndarray)
    assert y.shape == (10000,)
