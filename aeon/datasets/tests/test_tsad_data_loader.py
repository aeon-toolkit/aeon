"""Test anomaly detection dataset loaders."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aeon.datasets._tsad_data_loaders import (
    load_anomaly_detection,
    load_daphnet_s06r02e0,
    load_ecg_diff_count_3,
    load_from_timeeval_csv_file,
    load_kdd_tsad_135,
)
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
def test_load_anomaly_detection_wrong_name(mocker):
    """Test load non-existent anomaly detection datasets."""
    with pytest.raises(
        ValueError, match="The name of the dataset must be a tuple of two strings.*"
    ):
        load_anomaly_detection(("FOO", "BAR", "BAZ"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mocker.patch("aeon.datasets.tsad_datasets._DATA_FOLDER", tmp)
        with pytest.raises(
            ValueError,
            match="When loading a custom dataset, the extract_path must point "
            "to a TimeEval-formatted CSV file.*",
        ):
            load_anomaly_detection(("FOO", "BAR"))


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
def test_load_anomaly_detection_no_train_split(mocker):
    """Test load train split of anomaly detection dataset without training split."""
    name = ("Dodgers", "101-freeway-traffic")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mocker.patch("aeon.datasets.tsad_datasets._DATA_FOLDER", tmp)
        with pytest.raises(
            ValueError, match="Dataset .* does not have a training partition.*"
        ):
            load_anomaly_detection(name, extract_path=tmp, split="train")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
def test_load_anomaly_detection_from_archive(mocker):
    """Test load anomaly detection dataset from archive."""
    name = ("Genesis", "genesis-anomalies")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mocker.patch("aeon.datasets.tsad_datasets._DATA_FOLDER", tmp)
        X, y, meta = load_anomaly_detection(
            name, extract_path=tmp, return_metadata=True
        )
        assert isinstance(X, np.ndarray)
        assert X.shape == (16220, 18)
        assert isinstance(y, np.ndarray)
        assert y.shape == (16220,)
        assert isinstance(meta, dict)
        assert meta["learning_type"] == "unsupervised"
        assert meta["num_anomalies"] == 3
        np.testing.assert_almost_equal(meta["contamination"], 0.0031, decimal=4)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
def test_load_anomaly_detection_unavailable(mocker):
    """Test load anomaly detection dataset that is not available."""
    name = ("IOPS", "05f10d3a-239c-3bef-9bdc-a2feeb0037aa")
    with tempfile.TemporaryDirectory() as tmp:
        mocker.patch("aeon.datasets.tsad_datasets._DATA_FOLDER", Path(tmp))
        with pytest.raises(
            ValueError, match=f"Collection {name[0]} .* not available for download.*"
        ):
            load_anomaly_detection(name)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
def test_load_anomaly_detection_custom_file(mocker):
    """Test load anomaly detection dataset from custom file."""
    name = ("correlation-anomalies", "rmj-2-short-2-diff-channel")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mocker.patch("aeon.datasets.tsad_datasets._DATA_FOLDER", tmp)
        with pytest.raises(
            ValueError,
            match="When loading a custom dataset, the extract_path must point to a "
            "TimeEval-formatted CSV file.*",
        ):
            load_anomaly_detection(name, extract_path=tmp)

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
        Path(__file__).parent.parent / "data" / "Daphnet_S06R02E0" / "S06R02E0.csv"
    )
    assert isinstance(X, np.ndarray)
    assert X.shape == (7040, 9)
    assert isinstance(y, np.ndarray)
    assert y.shape == (7040,)


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


def test_load_daphnet_s06r02e0():
    """Test load Daphnet S06R02E0 dataset."""
    X, y = load_daphnet_s06r02e0()
    assert isinstance(X, np.ndarray)
    assert X.shape == (7040, 9)
    assert isinstance(y, np.ndarray)
    assert y.shape == (7040,)


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
