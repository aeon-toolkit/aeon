# -*- coding: utf-8 -*-
"""Unit tests for classifier deep learning base class functionality."""

import pickle

import pytest

from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["achieveordie"]


class _DummyDeepClassifierEmpty(BaseDeepClassifier):
    """Dummy Deep Classifier for testing empty base deep class save utilities."""

    def __init__(self):
        super(_DummyDeepClassifierEmpty, self).__init__()

    def build_model(self, input_shape, n_classes, **kwargs):
        return None

    def _fit(self, X, y):
        return self


class _DummyDeepClassifierFull(BaseDeepClassifier):
    """Dummy Deep Classifier to test serialization capabilities."""

    def __init__(
        self,
        optimizer,
    ):
        super(_DummyDeepClassifierFull, self).__init__()
        self.optimizer = optimizer

    def build_model(self, input_shape, n_classes, **kwargs):
        return None

    def _fit(self, X, y):
        return self


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_deep_estimator_empty():
    """Check if serialization works for empty dummy."""
    empty_dummy = _DummyDeepClassifierEmpty()
    serialized_empty = pickle.dumps(empty_dummy)
    deserialized_empty = pickle.loads(serialized_empty)
    assert empty_dummy.__dict__ == deserialized_empty.__dict__


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("optimizer", [None, "adam", "object-adamax"])
def test_deep_estimator_full(optimizer):
    """Check if serialization works for full dummy."""
    from tensorflow.keras.optimizers import Adamax, Optimizer, serialize

    if optimizer == "object-adamax":
        optimizer = Adamax()

    full_dummy = _DummyDeepClassifierFull(optimizer)
    serialized_full = pickle.dumps(full_dummy)
    deserialized_full = pickle.loads(serialized_full)

    if isinstance(optimizer, Optimizer):
        # assert same configuration of optimizer
        assert serialize(full_dummy.__dict__["optimizer"]) == serialize(
            deserialized_full.__dict__["optimizer"]
        )
        assert serialize(full_dummy.optimizer) == serialize(deserialized_full.optimizer)

        # assert weights of optimizers are same
        assert (
            full_dummy.optimizer.variables() == deserialized_full.optimizer.variables()
        )

        # remove optimizers from both to do full dict check,
        # since two different objects
        del full_dummy.__dict__["optimizer"]
        del deserialized_full.__dict__["optimizer"]

    # check if components are same
    assert full_dummy.__dict__ == deserialized_full.__dict__
