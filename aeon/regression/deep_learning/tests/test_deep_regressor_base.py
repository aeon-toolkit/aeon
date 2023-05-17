# -*- coding: utf-8 -*-
"""Unit tests for regressors deep learning base class functionality."""

import pytest

from aeon.regression.deep_learning.base import BaseDeepRegressor
from aeon.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["achieveordie", "hadifawaz1999"]


class _DummyDeepRegressorEmpty(BaseDeepRegressor):
    """Dummy Deep Regressor for testing empty base deep class save utilities."""

    def __init__(self):
        super(_DummyDeepRegressorEmpty, self).__init__()

    def build_model(self, input_shape, n_classes, **kwargs):
        return None

    def _fit(self, X, y):
        return self


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dummy_deep_regressor():
    import numpy as np

    dumpy_deep_clf = _DummyDeepRegressorEmpty()
    dumpy_deep_clf.build_model(input_shape=(10, 1), n_classes=2)
    dumpy_deep_clf.fit(
        X=np.random.normal(size=(10, 1, 10)), y=np.random.normal(size=(10,))
    )
