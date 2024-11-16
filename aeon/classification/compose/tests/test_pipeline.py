"""Unit tests for classification pipeline."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from aeon.classification import DummyClassifier
from aeon.classification.compose import ClassifierPipeline
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.dictionary_based import ContractableBOSS
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.mock_estimators import MockClassifier, MockCollectionTransformer
from aeon.transformations.collection import (
    AutocorrelationFunctionTransformer,
    HOG1DTransformer,
    Normalizer,
    Padder,
    Tabularizer,
)
from aeon.transformations.collection.feature_based import SevenNumberSummary


@pytest.mark.parametrize(
    "transformers",
    [
        Padder(pad_length=15),
        SevenNumberSummary(),
        [Padder(pad_length=15), Tabularizer(), StandardScaler()],
        [Padder(pad_length=15), SevenNumberSummary()],
        [Tabularizer(), StandardScaler(), SevenNumberSummary()],
        [
            Padder(pad_length=15),
            SevenNumberSummary(),
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

    y_pred = pipeline.predict(X_test)
    assert isinstance(y_pred, np.ndarray)

    if not isinstance(transformers, list):
        transformers = [transformers]

    for t in transformers:
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

    c.fit(X_train, y_train)
    assert_array_almost_equal(y_pred, c.predict(X_test))


@pytest.mark.parametrize(
    "transformers",
    [
        [Padder(pad_length=15), Tabularizer()],
        SevenNumberSummary(),
        [Tabularizer(), StandardScaler()],
        [Padder(pad_length=15), Tabularizer(), StandardScaler()],
        [Padder(pad_length=15), SevenNumberSummary()],
        [Tabularizer(), StandardScaler(), SevenNumberSummary()],
        [
            Padder(pad_length=15),
            SevenNumberSummary(),
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
    assert_array_almost_equal(y_pred, c.predict(X_test))


def test_unequal_tag_inference():
    """Test that ClassifierPipeline infers unequal length tag correctly."""
    X, y = make_example_3d_numpy_list(
        n_cases=10, min_n_timepoints=8, max_n_timepoints=12
    )

    t1 = SevenNumberSummary()
    t2 = Padder()
    t3 = Normalizer()
    t4 = AutocorrelationFunctionTransformer(n_lags=5)
    t5 = StandardScaler()
    t6 = Tabularizer()

    assert t1.get_tag("capability:unequal_length")
    assert t1.get_tag("output_data_type") == "Tabular"
    assert t2.get_tag("capability:unequal_length")
    assert t2.get_tag("removes_unequal_length")
    assert not t2.get_tag("output_data_type") == "Tabular"
    assert t3.get_tag("capability:unequal_length")
    assert not t3.get_tag("removes_unequal_length")
    assert not t3.get_tag("output_data_type") == "Tabular"
    assert not t4.get_tag("capability:unequal_length")

    c1 = DummyClassifier()
    c2 = MockClassifier()
    c3 = RandomForestClassifier(n_estimators=2)

    assert c1.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:unequal_length")

    # all handle unequal length
    p1 = ClassifierPipeline(transformers=t3, classifier=c1)
    assert p1.get_tag("capability:unequal_length")
    p1.fit(X, y)

    # classifier does not handle unequal length but transformer chain removes
    p2 = ClassifierPipeline(transformers=[t3, t2], classifier=c2)
    assert p2.get_tag("capability:unequal_length")
    p2.fit(X, y)

    # classifier does not handle unequal length but transformer chain removes (sklearn)
    p3 = ClassifierPipeline(transformers=[t3, t2, t6, t5], classifier=c3)
    assert p3.get_tag("capability:unequal_length")
    p3.fit(X, y)

    # transformers handle unequal length and output is tabular
    p4 = ClassifierPipeline(transformers=[t3, t1], classifier=c3)
    assert p4.get_tag("capability:unequal_length")
    p4.fit(X, y)

    # test they fit even if they cannot handle unequal length
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # transformers handle unequal length but classifier does not
    p5 = ClassifierPipeline(transformers=t3, classifier=c2)
    assert not p5.get_tag("capability:unequal_length")
    p5.fit(X, y)

    # classifier handles unequal length but transformer does not
    p6 = ClassifierPipeline(transformers=t4, classifier=c1)
    assert not p6.get_tag("capability:unequal_length")
    p6.fit(X, y)

    # transformer removes unequal length but prior cannot handle
    p7 = ClassifierPipeline(transformers=[t4, t2], classifier=c1)
    assert not p7.get_tag("capability:unequal_length")
    p7.fit(X, y)


def test_missing_tag_inference():
    """Test that ClassifierPipeline infers missing data tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    # tags are reset so this causes a crash due to t1
    # X[5, 0, 4] = np.nan

    t1 = MockCollectionTransformer()
    t1 = t1.set_tags(
        **{"capability:missing_values": True, "removes_missing_values": True}
    )
    t2 = Normalizer()
    t3 = StandardScaler()
    t4 = Tabularizer()

    assert t1.get_tag("capability:missing_values")
    assert t1.get_tag("removes_missing_values")
    assert not t2.get_tag("capability:missing_values")

    c1 = DummyClassifier()
    c2 = RocketClassifier(n_kernels=5)
    c3 = RandomForestClassifier(n_estimators=2)

    assert c1.get_tag("capability:missing_values")
    assert not c2.get_tag("capability:missing_values")

    # classifier does not handle missing values but transformer chain removes
    p1 = ClassifierPipeline(transformers=t1, classifier=c2)
    assert p1.get_tag("capability:missing_values")
    p1.fit(X, y)

    # classifier does not handle missing values but transformer chain removes (sklearn)
    p2 = ClassifierPipeline(transformers=[t1, t4, t3], classifier=c3)
    assert p2.get_tag("capability:missing_values")
    p2.fit(X, y)

    # test they fit even if they cannot handle missing data
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # transformers cannot handle missing data but classifier does
    p3 = ClassifierPipeline(transformers=t2, classifier=c1)
    assert not p3.get_tag("capability:missing_values")
    p3.fit(X, y)

    # transformers and classifier cannot handle missing data
    p4 = ClassifierPipeline(transformers=t2, classifier=c2)
    assert not p4.get_tag("capability:missing_values")
    p4.fit(X, y)

    # transformer removes missing values but prior cannot handle
    p5 = ClassifierPipeline(transformers=[t2, t1], classifier=c1)
    assert not p5.get_tag("capability:missing_values")
    p5.fit(X, y)


def test_multivariate_tag_inference():
    """Test that ClassifierPipeline infers multivariate tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=12)

    t1 = SevenNumberSummary()
    t2 = Normalizer()
    t3 = HOG1DTransformer()
    t4 = StandardScaler()

    assert t1.get_tag("capability:multivariate")
    assert t1.get_tag("output_data_type") == "Tabular"
    assert t2.get_tag("capability:multivariate")
    assert not t2.get_tag("output_data_type") == "Tabular"
    assert not t3.get_tag("capability:multivariate")

    c1 = DummyClassifier()
    c2 = ContractableBOSS(n_parameter_samples=5, max_ensemble_size=3)
    c3 = RandomForestClassifier(n_estimators=2)

    assert c1.get_tag("capability:multivariate")
    assert not c2.get_tag("capability:multivariate")

    # all handle multivariate
    p1 = ClassifierPipeline(transformers=t2, classifier=c1)
    assert p1.get_tag("capability:multivariate")
    p1.fit(X, y)

    # transformers handle multivariate and output is tabular
    p2 = ClassifierPipeline(transformers=[t1, t4], classifier=c3)
    assert p2.get_tag("capability:multivariate")
    p2.fit(X, y)

    # test they fit even if they cannot handle multivariate
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # transformers handle multivariate but classifier does not
    p3 = ClassifierPipeline(transformers=t2, classifier=c2)
    assert not p3.get_tag("capability:multivariate")
    p3.fit(X, y)

    # classifier handles multivariate but transformer does not
    p4 = ClassifierPipeline(transformers=t3, classifier=c1)
    assert not p4.get_tag("capability:multivariate")
    p4.fit(X, y)

    # transformer converts multivariate to tabular but prior cannot handle
    p5 = ClassifierPipeline(transformers=[t3, t1], classifier=c3)
    assert not p5.get_tag("capability:multivariate")
    p5.fit(X, y)
