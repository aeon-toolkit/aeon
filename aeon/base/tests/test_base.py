"""Tests for BaseAeonEstimator universal base class."""

import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._metadata_requests import MetadataRequest

from aeon.base import BaseAeonEstimator
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.classification.feature_based import SummaryClassifier
from aeon.testing.mock_estimators import MockClassifier
from aeon.testing.mock_estimators._mock_classifiers import (
    MockClassifierComposite,
    MockClassifierFullTags,
    MockClassifierParams,
)
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
from aeon.transformations.collection import Tabularizer


def test_reset():
    """Tests reset method for correct behaviour, on a simple estimator."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = MockClassifierParams(return_ones=True)
    clf.fit(X, y)

    assert clf.return_ones is True
    assert clf.value == 50
    assert clf.foo_ == "bar"
    assert clf.is_fitted is True
    clf.__secret_att = 42

    clf.reset()

    assert hasattr(clf, "return_ones") and clf.return_ones is True
    assert hasattr(clf, "value") and clf.value == 50
    assert hasattr(clf, "_tags") and clf._tags == MockClassifierParams._tags
    assert hasattr(clf, "is_fitted") and clf.is_fitted is False
    assert hasattr(clf, "__secret_att") and clf.__secret_att == 42
    assert hasattr(clf, "fit")
    assert not hasattr(clf, "foo_")

    clf.fit(X, y)
    clf.reset(keep="foo_")

    assert hasattr(clf, "is_fitted") and clf.is_fitted is False
    assert hasattr(clf, "foo_") and clf.foo_ == "bar"

    clf.fit(X, y)
    clf.random_att = 60
    clf.unwanted_att = 70
    clf.reset(keep=["foo_", "random_att"])

    assert hasattr(clf, "is_fitted") and clf.is_fitted is False
    assert hasattr(clf, "foo_") and clf.foo_ == "bar"
    assert hasattr(clf, "random_att") and clf.random_att == 60
    assert not hasattr(clf, "unwanted_att")


def test_reset_composite():
    """Test reset method for correct behaviour, on a composite estimator."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = MockClassifierComposite(mock=MockClassifierParams(return_ones=True))
    clf.fit(X, y)

    assert clf.foo_ == "bar"
    assert clf.mock_.foo_ == "bar"
    assert clf.mock.return_ones is True
    assert clf.mock_.return_ones is True

    clf.reset()

    assert hasattr(clf.mock, "return_ones") and clf.mock.return_ones is True
    assert not hasattr(clf, "mock_")
    assert not hasattr(clf, "foo_")
    assert not hasattr(clf.mock, "foo_")

    clf.fit(X, y)
    clf.reset(keep="mock_")

    assert not hasattr(clf, "foo_")
    assert hasattr(clf, "mock_")
    assert hasattr(clf.mock_, "foo_") and clf.mock_.foo_ == "bar"
    assert hasattr(clf.mock_, "return_ones") and clf.mock_.return_ones is True


def test_reset_invalid():
    """Tests that reset method raises error for invalid keep argument."""
    clf = MockClassifier()
    with pytest.raises(TypeError, match=r"keep must be a string or list"):
        clf.reset(keep=1)


def test_clone():
    """Tests that clone method correctly clones an estimator."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = MockClassifierParams(return_ones=True)
    clf.fit(X, y)

    clf_clone = clf.clone()
    assert clf_clone.return_ones is True
    assert not hasattr(clf_clone, "foo_")

    clf = SummaryClassifier(random_state=100)

    clf_clone = clf.clone(random_state=42)
    assert clf_clone.random_state == 1608637542


def test_clone_function():
    """Tests that _clone_estimator function correctly clones an estimator."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = MockClassifierParams(return_ones=True)
    clf.fit(X, y)

    clf_clone = _clone_estimator(clf)
    assert clf_clone.return_ones is True
    assert not hasattr(clf_clone, "foo_")

    clf = SummaryClassifier(random_state=100)

    clf_clone = _clone_estimator(clf, random_state=42)
    assert clf_clone.random_state == 1608637542


EXPECTED_MOCK_TAGS = {
    "X_inner_type": ["np-list", "numpy3D"],
    "algorithm_type": None,
    "cant_pickle": False,
    "capability:contractable": False,
    "capability:missing_values": True,
    "capability:multithreading": False,
    "capability:multivariate": True,
    "capability:train_estimate": False,
    "capability:unequal_length": True,
    "capability:univariate": True,
    "fit_is_empty": False,
    "non_deterministic": False,
    "python_dependencies": None,
    "python_version": None,
}


def test_get_class_tags():
    """Tests get_class_tags class method of BaseAeonEstimator for correctness."""
    child_tags = MockClassifierFullTags.get_class_tags()
    assert child_tags == EXPECTED_MOCK_TAGS


def test_get_class_tag():
    """Tests get_class_tag class method of BaseAeonEstimator for correctness."""
    for key in EXPECTED_MOCK_TAGS.keys():
        assert EXPECTED_MOCK_TAGS[key] == MockClassifierFullTags.get_class_tag(key)

    # these should be true for inherited class above, but false for the parent class
    assert BaseClassifier.get_class_tag("capability:missing_values") is False
    assert BaseClassifier.get_class_tag("capability:multivariate") is False
    assert BaseClassifier.get_class_tag("capability:unequal_length") is False

    assert (
        BaseAeonEstimator.get_class_tag(
            "invalid_tag", raise_error=False, tag_value_default=50
        )
        == 50
    )

    with pytest.raises(ValueError, match=r"Tag with name invalid_tag"):
        BaseAeonEstimator.get_class_tag("invalid_tag")


def test_get_tags():
    """Tests get_tags method of BaseAeonEstimator for correctness."""
    child_tags = MockClassifierFullTags().get_tags()
    assert child_tags == EXPECTED_MOCK_TAGS


def test_get_tag():
    """Tests get_tag method of BaseAeonEstimator for correctness."""
    clf = MockClassifierFullTags()
    for key in EXPECTED_MOCK_TAGS.keys():
        assert EXPECTED_MOCK_TAGS[key] == clf.get_tag(key)

    # these should be true for class above which overrides, but false for this which
    # does not
    clf = MockClassifier()
    assert clf.get_tag("capability:missing_values") is False
    assert clf.get_tag("capability:multivariate") is False
    assert clf.get_tag("capability:unequal_length") is False

    assert clf.get_tag("invalid_tag", raise_error=False, tag_value_default=50) == 50

    with pytest.raises(ValueError, match=r"Tag with name invalid_tag"):
        clf.get_tag("invalid_tag")


def test_set_tags():
    """Tests set_tags method of BaseAeonEstimator for correctness."""
    clf = MockClassifier()

    tags_to_set = {
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:unequal_length": True,
    }
    clf.set_tags(**tags_to_set)

    assert clf.get_tag("capability:missing_values") is True
    assert clf.get_tag("capability:multivariate") is True
    assert clf.get_tag("capability:unequal_length") is True

    clf.reset()

    assert clf.get_tag("capability:missing_values") is False
    assert clf.get_tag("capability:multivariate") is False
    assert clf.get_tag("capability:unequal_length") is False


def test_get_fitted_params():
    """Tests fitted parameter retrieval."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    non_composite = MockClassifier()
    non_composite.fit(X, y)
    composite = MockClassifierComposite()
    composite.fit(X, y)

    params = non_composite.get_fitted_params()
    comp_params = composite.get_fitted_params()

    expected = {
        "foo_",
        "classes_",
        "metadata_",
        "n_classes_",
    }

    assert isinstance(params, dict)
    assert set(params.keys()) == expected
    assert params["foo_"] is composite.foo_

    assert isinstance(comp_params, dict)
    assert set(comp_params.keys()) == expected.union(
        {
            "mock_",
            "mock___classes_",
            "mock___foo_",
            "mock___metadata_",
            "mock___n_classes_",
        }
    )
    assert comp_params["foo_"] is composite.foo_
    assert comp_params["mock___foo_"] is composite.mock_.foo_

    params_shallow = non_composite.get_fitted_params(deep=False)
    comp_params_shallow = composite.get_fitted_params(deep=False)

    assert isinstance(params_shallow, dict)
    assert set(params_shallow.keys()) == set(params.keys())

    assert isinstance(comp_params_shallow, dict)
    assert set(comp_params_shallow.keys()) == set(params.keys()).union({"mock_"})


def test_get_fitted_params_sklearn():
    """Tests fitted parameter retrieval with sklearn components."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = SummaryClassifier(estimator=DecisionTreeClassifier())
    clf.fit(X, y)

    params = clf.get_fitted_params()

    assert "estimator_" in params.keys()
    assert "transformer_" in params.keys()
    assert "estimator___tree_" in params.keys()
    assert "estimator___max_features_" in params.keys()

    # pipeline
    pipe = make_pipeline(Tabularizer(), StandardScaler(), DecisionTreeClassifier())
    clf = SummaryClassifier(estimator=pipe)
    clf.fit(X, y)

    params = clf.get_fitted_params()

    assert "estimator_" in params.keys()
    assert "transformer_" in params.keys()


def test_check_is_fitted():
    """Test _check_is_fitted works correctly."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = MockClassifier()

    with pytest.raises(ValueError, match=r"has not been fitted yet"):
        clf._check_is_fitted()

    clf.fit(X, y)

    clf._check_is_fitted()


def test_create_test_instance():
    """Test _create_test_instance works as expected."""
    clf = SummaryClassifier._create_test_instance()

    assert isinstance(clf, SummaryClassifier)
    assert clf.estimator.n_estimators == 2


def test_overridden_sklearn():
    """Tests that overridden sklearn components return expected outputs."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = MockClassifier()
    clf.fit(X, y)

    assert clf.__sklearn_is_fitted__() == clf.is_fitted

    assert isinstance(clf._get_default_requests(), MetadataRequest)

    with pytest.raises(NotImplementedError):
        clf._validate_data()

    with pytest.raises(NotImplementedError):
        clf.get_metadata_routing()
