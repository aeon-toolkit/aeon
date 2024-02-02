"""Functions to test the functions in experiments.py."""

import os
import tempfile

import pytest

from aeon.benchmarking.experiments import (
    run_classification_experiment,
    run_clustering_experiment,
)
from aeon.classification import DummyClassifier
from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import load_unit_test
from aeon.testing.test_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_run_clustering_experiment():
    """Test running and saving results for clustering.

    Currently it just checks the files have been created, then deletes them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dataset = "UnitTest"
        train_X, train_Y = load_unit_test("TRAIN")
        test_X, test_Y = load_unit_test("TEST")
        run_clustering_experiment(
            train_X,
            TimeSeriesKMeans(n_clusters=2),
            results_path=tmp,
            trainY=train_Y,
            testX=test_X,
            testY=test_Y,
            cls_name="kmeans",
            dataset_name=dataset,
            resample_id=0,
        )
        test_path = os.path.join(tmp, f"kmeans/Predictions/{dataset}/testResample0.csv")
        train_path = os.path.join(
            tmp, f"kmeans/Predictions/{dataset}/trainResample0.csv"
        )
        assert os.path.isfile(test_path)
        assert os.path.isfile(train_path)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
def test_run_classification_experiment(tmp_path):
    """Test running and saving results for classifiers.

    Currently it just checks the files have been created, then deletes them.
    """

    with tempfile.TemporaryDirectory() as tmp:
        dataset = "UnitTest"
        train_X, train_Y = load_unit_test("TRAIN")
        test_X, test_Y = load_unit_test("TEST")
        run_classification_experiment(
            train_X,
            train_Y,
            test_X,
            test_Y,
            DummyClassifier(),
            tmp,
            cls_name="DummyClassifier",
            dataset="UnitTest",
            resample_id=0,
            train_file=True,
        )
        test_path = os.path.join(
            tmp, f"DummyClassifier/Predictions/{dataset}/testResample0.csv"
        )
        train_path = os.path.join(
            tmp, f"DummyClassifier/Predictions/{dataset}/trainResample0.csv"
        )
        assert os.path.isfile(test_path)
        assert os.path.isfile(train_path)
