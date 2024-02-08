"""Unit tests for (dunder) composition functionality attached to the base class."""

__author__ = ["fkiraly", "TonyBagnall"]

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from aeon.classification import DummyClassifier
from aeon.classification.compose import ClassifierPipeline
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.dictionary_based import ContractableBOSS
from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.testing.utils.estimator_checks import _assert_array_almost_equal
from aeon.transformations.adapt import TabularToSeriesAdaptor
from aeon.transformations.collection import PaddingTransformer, Tabularizer
from aeon.transformations.collection.feature_based import SevenNumberSummaryTransformer
from aeon.transformations.impute import Imputer


@pytest.mark.parametrize(
    "transformers",
    [
        PaddingTransformer(pad_length=15),
        SevenNumberSummaryTransformer(),
        TabularToSeriesAdaptor(StandardScaler()),
        [PaddingTransformer(pad_length=15), Tabularizer(), StandardScaler()],
        [PaddingTransformer(pad_length=15), SevenNumberSummaryTransformer()],
        [Tabularizer(), StandardScaler(), SevenNumberSummaryTransformer()],
        [
            TabularToSeriesAdaptor(StandardScaler()),
            PaddingTransformer(pad_length=15),
            SevenNumberSummaryTransformer(),
        ],
    ],
)
def test_classifier_pipeline(transformers):
    """Test the classifier pipeline."""
    X_train, y_train = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    c = DummyClassifier()
    pipeline = ClassifierPipeline(transformers=transformers, classifier=c)
    pipeline.fit(X_train, y_train)
    c.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    assert isinstance(y_pred, np.ndarray)

    if not isinstance(transformers, list):
        transformers = [transformers]

    for t in transformers:
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

    c.fit(X_train, y_train)
    _assert_array_almost_equal(y_pred, c.predict(X_test))


@pytest.mark.parametrize(
    "transformers",
    [
        [PaddingTransformer(pad_length=15), Tabularizer()],
        SevenNumberSummaryTransformer(),
        [Tabularizer(), StandardScaler()],
        [PaddingTransformer(pad_length=15), Tabularizer(), StandardScaler()],
        [PaddingTransformer(pad_length=15), SevenNumberSummaryTransformer()],
        [Tabularizer(), StandardScaler(), SevenNumberSummaryTransformer()],
        [
            TabularToSeriesAdaptor(StandardScaler()),
            PaddingTransformer(pad_length=15),
            SevenNumberSummaryTransformer(),
        ],
    ],
)
def test_sklearn_classifier_pipeline(transformers):
    """Test classifier pipeline with sklearn estimator."""
    X_train, y_train = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    c = RandomForestClassifier(n_estimators=2, random_state=0)
    pipeline = ClassifierPipeline(transformers=transformers, classifier=c)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    assert isinstance(y_pred, np.ndarray)

    if not isinstance(transformers, list):
        transformers = [transformers]

    for t in transformers:
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

    c.fit(X_train, y_train)
    _assert_array_almost_equal(y_pred, c.predict(X_test))


def test_tag_inference():
    """Test that ClassifierPipeline infers tags correctly."""
    t1 = SevenNumberSummaryTransformer()
    t2 = PaddingTransformer()
    t3 = Imputer()
    c1 = DummyClassifier()
    c2 = RocketClassifier(num_kernels=5)
    c3 = ContractableBOSS(n_parameter_samples=5, max_ensemble_size=3)

    assert c1.get_tag("capability:unequal_length")
    assert c1.get_tag("capability:missing_values")
    assert c1.get_tag("capability:multivariate")
    assert not c2.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:missing_values")
    assert not c3.get_tag("capability:multivariate")

    assert ClassifierPipeline(transformers=t1, classifier=c1).get_tag(
        "capability:unequal_length"
    )
    assert ClassifierPipeline(transformers=[t1, t2], classifier=c2).get_tag(
        "capability:unequal_length"
    )
    assert not ClassifierPipeline(transformers=t1, classifier=c2).get_tag(
        "capability:unequal_length"
    )

    assert ClassifierPipeline(transformers=[t1, t3], classifier=c1).get_tag(
        "capability:missing_values"
    )
    assert ClassifierPipeline(transformers=[t1, t3], classifier=c2).get_tag(
        "capability:missing_values"
    )
    assert not ClassifierPipeline(transformers=t1, classifier=c1).get_tag(
        "capability:missing_values"
    )

    assert ClassifierPipeline(transformers=t1, classifier=c1).get_tag(
        "capability:multivariate"
    )
    assert not ClassifierPipeline(transformers=t1, classifier=c3).get_tag(
        "capability:multivariate"
    )
