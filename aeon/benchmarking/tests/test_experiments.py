"""Functions to test the functions in experiments.py."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from aeon.benchmarking.experiments import (
    run_classification_experiment,
    run_clustering_experiment,
    stratified_resample,
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


INPUT_TYPES = ["numpy3D", "np-list", "pd.DataFrame"]


@pytest.mark.parametrize("input_type", INPUT_TYPES)
def test_stratified_resample(input_type):
    """Test Stratified resampling."""
    random_state = np.random.RandomState(0)
    if input_type == "numpy3D":
        X_train = random_state.random((10, 1, 100))
        X_test = random_state.random((10, 1, 100))
    elif input_type == "np-list":
        X_train = [random_state.random((1, 100)) for _ in range(10)]
        X_test = [random_state.random((1, 100)) for _ in range(10)]
    else:
        train_series = [pd.Series(random_state.random(100)) for _ in range(10)]
        test_series = [pd.Series(random_state.random(100)) for _ in range(10)]
        X_train = pd.DataFrame({"dim_0": train_series})
        X_test = pd.DataFrame({"dim_0": test_series})
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_test = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    new_X_train, new_y_train, new_X_test, new_y_test = stratified_resample(
        X_train, y_train, X_test, y_test, random_state
    )

    # Valid return type
    assert type(X_train) is type(new_X_train) and type(X_test) is type(new_X_test)

    classes_train, classes_count_train = np.unique(y_train, return_counts=True)
    classes_test, classes_count_test = np.unique(y_test, return_counts=True)
    classes_new_train, classes_count_new_train = np.unique(
        new_y_train, return_counts=True
    )
    classes_new_test, classes_count_new_test = np.unique(new_y_test, return_counts=True)

    # Assert same class distributions
    assert np.all(classes_train == classes_new_train)
    assert np.all(classes_count_train == classes_count_new_train)
    assert np.all(classes_test == classes_new_test)
    assert np.all(classes_count_test == classes_count_new_test)
