"""Test anomaly detection dataset loaders."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeon.datasets._tsad_data_loaders import (
    load_anomaly_detection,
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
        assert X.flags.writeable
        assert isinstance(y, np.ndarray)
        assert y.shape == (16220,)
        assert y.flags.writeable
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
def test_load_anomaly_detection_custom_file(mocker, tmp_path):
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

    for learning_type in ["unsupervised", "semi-supervised", "supervised"]:
        path = tmp_path / f"{learning_type}.csv"
        labels = [0, 0, 0, 0] if learning_type == "semi-supervised" else [0, 1, 0, 1]
        pd.DataFrame(
            {
                "timestamp": range(4),
                "value": [1.0, 2.0, 3.0, 4.0],
                "is_anomaly": labels,
            }
        ).to_csv(path, index=False)
        X, y, meta = load_anomaly_detection(
            name,
            extract_path=path,
            split="test" if learning_type == "unsupervised" else "train",
            return_metadata=True,
        )
        assert X.shape[0] == y.shape[0] == 4
        assert meta["learning_type"] == learning_type
        assert meta["num_anomalies"] == sum(labels)


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
    assert X.flags.writeable
    assert isinstance(y, np.ndarray)
    assert y.shape == (7501,)
    assert y.flags.writeable


def test_load_from_timeeval_csv_file_multivariate(tmp_path):
    """Test multivariate CSV loading without a bundled dataset."""
    path = tmp_path / "multivariate.csv"
    values = np.arange(36).reshape(4, 9)
    frame = pd.DataFrame(values, columns=[f"channel_{i}" for i in range(9)])
    frame.insert(0, "timestamp", range(4))
    frame["is_anomaly"] = [0, 1, 0, 0]
    frame.to_csv(path, index=False)
    X, y = load_from_timeeval_csv_file(path)
    np.testing.assert_array_equal(X, values)
    np.testing.assert_array_equal(y, [0, 1, 0, 0])
    assert X.flags.writeable
    assert y.flags.writeable


def test_load_kdd_tsad_135():
    """Test load KDD TSAD 135 dataset."""
    X, y = load_kdd_tsad_135()
    assert isinstance(X, np.ndarray)
    assert X.shape == (7501,)
    assert X.flags.writeable
    assert isinstance(y, np.ndarray)
    assert y.shape == (7501,)
    assert y.flags.writeable

    X_train, y_train = load_kdd_tsad_135(split="train")
    assert isinstance(X_train, np.ndarray)
    assert X_train.shape == (1200,)
    assert X_train.flags.writeable
    assert isinstance(y_train, np.ndarray)
    assert y_train.shape == (1200,)
    assert y_train.flags.writeable
