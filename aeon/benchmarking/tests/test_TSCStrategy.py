# -*- coding: utf-8 -*-
"""Data storage for benchmarking."""
import pytest

from aeon.benchmarking.strategies import TSCStrategy
from aeon.benchmarking.tasks import TSCTask
from aeon.classification.compose import ComposableTimeSeriesForestClassifier
from aeon.datasets import load_gunpoint, load_italy_power_demand

classifier = ComposableTimeSeriesForestClassifier(n_estimators=2)

DATASET_LOADERS = (load_gunpoint, load_italy_power_demand)


# Test output of time-series classification strategies
@pytest.mark.parametrize("dataset", DATASET_LOADERS)
def test_TSCStrategy(dataset):
    """Test strategy."""
    train = dataset(split="train", return_X_y=False, return_type="nested_univ")
    test = dataset(split="test", return_X_y=False, return_type="nested_univ")
    s = TSCStrategy(classifier)
    task = TSCTask(target="class_val")
    s.fit(task, train)
    y_pred = s.predict(test)
    assert y_pred.shape == test[task.target].shape
