# -*- coding: utf-8 -*-
"""Classifier Base."""
__all__ = [
    "BaseClassifier",
    "DummyClassifier",
]

from sktime.classification._dummy import DummyClassifier
from sktime.classification.base import BaseClassifier
