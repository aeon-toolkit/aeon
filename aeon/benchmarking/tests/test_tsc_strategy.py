# -*- coding: utf-8 -*-
"""Data storage for benchmarking."""
from aeon.benchmarking.strategies import TSCStrategy
from aeon.benchmarking.tasks import TSCTask
from aeon.classification import DummyClassifier
from aeon.datasets import load_unit_test

classifier = DummyClassifier()


def test_TSCStrategy():
    """Test strategy."""
    train = load_unit_test(split="train", return_X_y=False, return_type="nested_univ")
    test = load_unit_test(split="test", return_X_y=False, return_type="nested_univ")
    s = TSCStrategy(classifier)
    task = TSCTask(target="class_val")
    s.fit(task, train)
    y_pred = s.predict(test)
    assert y_pred.shape == test[task.target].shape
