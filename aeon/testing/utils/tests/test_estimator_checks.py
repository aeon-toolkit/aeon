"""Tests for estimator checking utilities."""

import numpy as np
import pytest

from aeon.testing.mock_estimators import MockClassifier, MockClassifierPredictProba
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import (
    _assert_predict_labels,
    _assert_predict_probabilities,
    _changed_state,
    _get_tag,
    _holds_deep_learning_state,
    _run_estimator_method,
    _snapshot_state,
)

DATATYPE = "EqualLengthUnivariate-Classification-numpy3D"
N_CASES = 5
N_CLASSES = 2


class _Unpickleable:
    """Object which cannot be pickled, and so cannot be hashed.

    ``pickleable`` allows an object to only become unpickleable after a state
    snapshot has been taken.
    """

    def __init__(self, pickleable=False):
        self.pickleable = pickleable

    def __reduce__(self):
        if not self.pickleable:
            raise TypeError("cannot pickle")
        return (type(self), ())


class _FrameworkObject:
    """Stand-in for a ``keras`` object.

    ``__module__`` is what marks the object as framework state. ``__reduce__``
    keeps it hashable, as the faked module cannot be imported to unpickle it.
    """

    __module__ = "keras.src.models"

    def __reduce__(self):
        return (dict, ())


def test_run_estimator_method_runs_methods():
    """Test that methods are run with the correct test data."""
    estimator = MockClassifierPredictProba()

    assert _run_estimator_method(estimator, "fit", DATATYPE, "train") is estimator

    y_pred = _run_estimator_method(estimator, "predict", DATATYPE, "test")
    y_proba = _run_estimator_method(estimator, "predict_proba", DATATYPE, "test")

    assert y_pred.shape == (N_CASES,)
    assert y_proba.shape == (N_CASES, N_CLASSES)


def test_run_estimator_method_raises_for_missing_soft_dependency():
    """Test that a ModuleNotFoundError is reported as a soft dependency issue."""

    class _MissingDependency(MockClassifier):
        def _fit(self, X, y):
            raise ModuleNotFoundError("no soft dependency")

    with pytest.raises(RuntimeError, match="python_dependencies"):
        _run_estimator_method(_MissingDependency(), "fit", DATATYPE, "train")


def test_get_tag_for_class_instance_and_none():
    """Test that tags are found for both classes and instances."""
    estimator = MockClassifier().set_tags(algorithm_type="feature")

    assert _get_tag(MockClassifier, "algorithm_type") is None
    assert _get_tag(MockClassifier(), "algorithm_type") is None
    assert _get_tag(estimator, "algorithm_type") == "feature"
    assert _get_tag(None, "algorithm_type") is None


def test_get_tag_unknown_tag():
    """Test that unknown tags return the default or raise."""
    assert _get_tag(MockClassifier(), "not_a_tag", default="default") == "default"

    with pytest.raises(ValueError):
        _get_tag(MockClassifier(), "not_a_tag", raise_error=True)


def test_assert_predict_labels():
    """Test that valid predictions pass and invalid predictions are caught."""
    y_pred = np.zeros(N_CASES)

    _assert_predict_labels(y_pred, DATATYPE)
    _assert_predict_labels(y_pred, FULL_TEST_DATA_DICT[DATATYPE]["test"][0])
    _assert_predict_labels(y_pred, DATATYPE, unique_labels=[0])

    with pytest.raises(AssertionError):
        _assert_predict_labels(list(y_pred), DATATYPE)

    with pytest.raises(AssertionError):
        _assert_predict_labels(np.zeros(N_CASES + 1), DATATYPE)

    with pytest.raises(AssertionError):
        _assert_predict_labels(y_pred, DATATYPE, unique_labels=[1])


def test_assert_predict_probabilities():
    """Test that valid probabilities pass and invalid probabilities are caught."""
    y_proba = np.full((N_CASES, N_CLASSES), 0.5)

    _assert_predict_probabilities(y_proba, DATATYPE)
    _assert_predict_probabilities(
        y_proba, FULL_TEST_DATA_DICT[DATATYPE]["test"][0], n_classes=N_CLASSES
    )

    with pytest.raises(AssertionError):
        _assert_predict_probabilities(y_proba, DATATYPE, n_classes=N_CLASSES + 1)

    # probabilities outside of [0, 1]
    with pytest.raises(AssertionError):
        _assert_predict_probabilities(
            np.full((N_CASES, N_CLASSES), [2.0, -1.0]), DATATYPE
        )

    # probabilities which do not sum to 1
    with pytest.raises(AssertionError):
        _assert_predict_probabilities(np.full((N_CASES, N_CLASSES), 0.25), DATATYPE)


def test_assert_predict_probabilities_requires_n_classes():
    """Test that n_classes is required when not using a test dataset string."""
    with pytest.raises(ValueError, match="n_classes must be provided"):
        _assert_predict_probabilities(
            np.full((N_CASES, N_CLASSES), 0.5),
            FULL_TEST_DATA_DICT[DATATYPE]["test"][0],
        )


def test_holds_deep_learning_state_finds_nested_objects():
    """Test that framework objects held inside a container are found."""
    obj = _FrameworkObject()
    estimator = MockClassifier()
    estimator.model_ = obj

    assert _holds_deep_learning_state(obj)
    assert _holds_deep_learning_state([obj])
    assert _holds_deep_learning_state({"model": obj})
    assert _holds_deep_learning_state(estimator)
    # i.e. a list of sub-estimators, each holding a framework model
    assert _holds_deep_learning_state([estimator])


def test_holds_deep_learning_state_ignores_ordinary_values():
    """Test that values without framework state are not treated as opaque."""
    assert not _holds_deep_learning_state("foo")
    assert not _holds_deep_learning_state(42)
    assert not _holds_deep_learning_state(np.zeros(3))
    assert not _holds_deep_learning_state({"values": [1, 2]})
    assert not _holds_deep_learning_state(MockClassifier())


def test_snapshot_state_detects_nested_mutation():
    """Test that hashes detect mutation inside a top-level attribute."""
    estimator = MockClassifier()
    estimator.state = {"values": [1, 2]}

    before = _snapshot_state(estimator)

    assert before["state"][0] == "hash"
    assert _changed_state(before, estimator) == set()

    estimator.state["values"].append(3)

    assert _changed_state(before, estimator) == {"state"}


def test_snapshot_state_falls_back_to_identity():
    """Test identity fallback and replacement detection for unpickleable state."""
    estimator = MockClassifier()
    estimator.state = _Unpickleable()

    before = _snapshot_state(estimator)

    assert before["state"] == ("identity", estimator.state)
    assert _changed_state(before, estimator) == set()

    estimator.state = _Unpickleable()

    assert _changed_state(before, estimator) == {"state"}


def test_snapshot_state_falls_back_per_attribute():
    """Test that only the attributes which cannot be hashed use identity."""
    estimator = MockClassifier()
    estimator.hashable = {"values": [1, 2]}
    estimator.unhashable = _Unpickleable()

    before = _snapshot_state(estimator)

    assert before["hashable"][0] == "hash"
    assert before["unhashable"][0] == "identity"

    # nested changes are still detected for the attributes which can be hashed
    estimator.hashable["values"].append(3)

    assert _changed_state(before, estimator) == {"hashable"}


def test_snapshot_state_uses_identity_for_deep_learning_state():
    """Test that deep learning framework state uses identity, but nothing else."""
    estimator = MockClassifier().set_tags(algorithm_type="deeplearning")
    estimator.model_ = _FrameworkObject()
    estimator.state = {"values": [1, 2]}

    before = _snapshot_state(estimator)

    assert before["model_"] == ("identity", estimator.model_)
    assert before["state"][0] == "hash"

    # every attribute which is not framework state is still checked
    estimator.state["values"].append(3)

    assert _changed_state(before, estimator) == {"state"}


def test_snapshot_state_deep_learning_state_ignored_for_other_estimators():
    """Test that framework state is only exempt for deep learning estimators."""
    estimator = MockClassifier()
    estimator.model_ = _FrameworkObject()

    before = _snapshot_state(estimator)

    assert before["model_"][0] == "hash"


def test_changed_state_detects_added_and_removed_attributes():
    """Test that changes to the set of estimator attributes are detected."""
    estimator = MockClassifier()
    estimator.state = [1, 2, 3]

    before = _snapshot_state(estimator)

    del estimator.state
    estimator.added = "new"

    assert _changed_state(before, estimator) == {"added", "state"}


def test_changed_state_detects_value_becoming_unpickleable():
    """Test that failure to hash a previously hashable value counts as a change."""
    estimator = MockClassifier()
    estimator.state = _Unpickleable(pickleable=True)

    before = _snapshot_state(estimator)
    assert before["state"][0] == "hash"

    estimator.state.pickleable = False

    assert _changed_state(before, estimator) == {"state"}
