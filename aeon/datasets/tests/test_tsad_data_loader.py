"""Test anomaly detection dataset loaders."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from aeon.datasets._tsad_data_loaders import (
    _DATA_FOLDER,
    load_anomaly_detection,
    load_calit2,
    load_dodgers,
    load_from_timeeval_csv_file,
    load_rmj_2_short_2_diff_channel,
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


def test_load_anomaly_detection_from_repo():
    """Test load univariate anomaly detection dataset from repo."""
    name = ("Dodgers", "101-freeway-traffic")
    X, y, meta = load_anomaly_detection(name, return_metadata=True)
    assert isinstance(X, np.ndarray)
    assert X.shape == (50400,)
    assert isinstance(y, np.ndarray)
    assert y.shape == (50400,)
    assert isinstance(meta, dict)
    assert meta["learning_type"] == "unsupervised"
    assert meta["num_anomalies"] == 133
    np.testing.assert_almost_equal(meta["contamination"], 0.1113, decimal=4)


def test_load_anomaly_detection_from_repo_multivariate():
    """Test load multivariate anomaly detection dataset from repo."""
    name = ("CalIt2", "CalIt2-traffic")
    X, y, meta = load_anomaly_detection(name, return_metadata=True)
    assert isinstance(X, np.ndarray)
    assert X.shape == (5040, 2)
    assert isinstance(y, np.ndarray)
    assert y.shape == (5040,)
    assert isinstance(meta, dict)
    assert meta["learning_type"] == "unsupervised"
    assert meta["num_anomalies"] == 29
    np.testing.assert_almost_equal(meta["contamination"], 0.0409, decimal=4)


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
        assert X.shape[1] == 2
        assert meta["learning_type"] == learning_type
        assert meta["num_anomalies"] == 0 if learning_type == "semi-supervised" else 2

    _test("rmj-2-short-2-diff-channel.test.csv", "unsupervised")
    _test("rmj-2-short-2-diff-channel.train_no_anomaly.csv", "semi-supervised")
    _test("rmj-2-short-2-diff-channel.train_anomaly.csv", "supervised")


def test_load_from_timeeval_csv_file_univariate():
    """Test load univariate dataset from file."""
    X, y = load_from_timeeval_csv_file(
        _DATA_FOLDER / "univariate" / "Dodgers" / "101-freeway-traffic.test.csv"
    )
    assert isinstance(X, np.ndarray)
    assert X.shape == (50400,)
    assert isinstance(y, np.ndarray)
    assert y.shape == (50400,)


def test_load_from_timeeval_csv_file_multivariate():
    """Test load multivariate dataset from file."""
    X, y = load_from_timeeval_csv_file(
        _DATA_FOLDER / "multivariate" / "CalIt2" / "CalIt2-traffic.test.csv"
    )
    assert isinstance(X, np.ndarray)
    assert X.shape == (5040, 2)
    assert isinstance(y, np.ndarray)
    assert y.shape == (5040,)


def test_load_dodgers():
    """Test load Dodgers dataset."""
    X, y = load_dodgers()
    assert isinstance(X, np.ndarray)
    assert X.shape == (50400,)
    assert isinstance(y, np.ndarray)
    assert y.shape == (50400,)


def test_load_calit2():
    """Test load CalIt2 dataset."""
    X, y = load_calit2()
    assert isinstance(X, np.ndarray)
    assert X.shape == (5040, 2)
    assert isinstance(y, np.ndarray)
    assert y.shape == (5040,)


@pytest.mark.parametrize(
    "learning_type", ["unsupervised", "semi-supervised", "supervised"]
)
def test_load_rmj_2_short_2_diff_channel(learning_type):
    """Test load rmj-2-short-2-diff-channel dataset."""
    if learning_type == "unsupervised":
        X, y = load_rmj_2_short_2_diff_channel(learning_type)
    else:
        X, y, X_train, y_train = load_rmj_2_short_2_diff_channel(learning_type)
        assert isinstance(X_train, np.ndarray)
        assert X_train.shape == (10000, 2)
        assert isinstance(y_train, np.ndarray)
        assert y_train.shape == (10000,)
        if learning_type == "semi-supervised":
            assert np.sum(y_train) == 0

    assert isinstance(X, np.ndarray)
    assert X.shape == (10000, 2)
    assert isinstance(y, np.ndarray)
    assert y.shape == (10000,)
