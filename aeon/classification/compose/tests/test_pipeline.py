# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for composition functionality attached to the base class."""

__author__ = ["fkiraly", "TonyBagnall"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from aeon.classification import DummyClassifier
from aeon.classification.compose import ClassifierPipeline, SklearnClassifierPipeline
from aeon.classification.convolution_based import RocketClassifier
from aeon.transformations.panel.pad import PaddingTransformer
from aeon.transformations.series.exponent import ExponentTransformer
from aeon.transformations.series.impute import Imputer
from aeon.utils._testing.collection import make_3d_test_data


def test_classifier_pipeline():
    """Test the classifier pipeline."""
    RAND_SEED = 42
    X, y = make_3d_test_data(n_cases=10, n_timepoints=20, random_state=RAND_SEED)

    X_test, _ = make_3d_test_data(n_cases=10, n_timepoints=20, random_state=RAND_SEED)

    t1 = ExponentTransformer(power=4)
    t2 = ExponentTransformer(power=0.25)
    c = DummyClassifier()
    pipeline = ClassifierPipeline(transformers=[t1, t2], classifier=c)
    y_pred = pipeline.fit(X, y).predict(X_test)
    assert isinstance(y_pred, np.ndarray)


def test_sklearn_classifier_pipeline():
    """Test auto-adapter for sklearn in mul."""
    RAND_SEED = 42
    X, y = make_3d_test_data(n_cases=10, n_timepoints=20, random_state=RAND_SEED)

    X_test, _ = make_3d_test_data(n_cases=10, n_timepoints=20, random_state=RAND_SEED)
    t1 = ExponentTransformer(power=2)
    t2 = StandardScaler()
    c = RandomForestClassifier(n_estimators=2)
    pipeline = SklearnClassifierPipeline(transformers=[t1, t2], classifier=c)

    y_pred = pipeline.fit(X, y).predict(X_test)
    assert isinstance(y_pred, np.ndarray)


def test_missing_unequal_tag_inference():
    """Test that ClassifierPipeline infers missing/unequal tags correctly."""
    c = RocketClassifier(num_kernels=100)
    t1 = ExponentTransformer(power=4)
    t2 = PaddingTransformer()
    t3 = Imputer()
    c1 = ClassifierPipeline(transformers=[t1, t2], classifier=c)
    c2 = ClassifierPipeline(transformers=[t1, t1], classifier=c)
    c3 = ClassifierPipeline(transformers=[t3, t1], classifier=c)
    c4 = ClassifierPipeline(transformers=[t1, t3], classifier=c)

    assert c1.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:unequal_length")
    assert c3.get_tag("capability:missing_values")
    assert not c4.get_tag("capability:missing_values")
