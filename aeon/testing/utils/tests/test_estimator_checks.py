"""Tests for estimator checking utilities."""

import numpy as np

from aeon.testing.mock_estimators import MockClassifier
from aeon.testing.utils.estimator_checks import (
    _changed_state,
    _holds_deep_learning_state,
    _snapshot_state,
)


class _Unpickleable:
    """Object which cannot be hashed through pickle."""

    def __reduce__(self):
        raise TypeError("cannot pickle")


class _ConditionallyUnpickleable:
    """Object which can become unpickleable after the state snapshot."""

    def __init__(self):
        self.pickleable = True

    def __reduce__(self):
        if not self.pickleable:
            raise TypeError("cannot pickle")
        return (type(self), ())


class _FrameworkObject:
    """Stand-in for a deep learning framework object.

    ``__module__`` is what marks the object as framework state. ``__reduce__``
    keeps it hashable, as the faked module cannot be imported to unpickle it.
    """

    __module__ = "keras.src.models"

    def __reduce__(self):
        return (dict, ())


class _Holder:
    """Object holding another object as an attribute."""

    def __init__(self, value):
        self.value = value


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


def test_holds_deep_learning_state_finds_nested_objects():
    """Test that framework objects held inside a container are found."""
    obj = _FrameworkObject()

    assert _holds_deep_learning_state(obj)
    assert _holds_deep_learning_state([obj])
    assert _holds_deep_learning_state({"model": obj})
    assert _holds_deep_learning_state(_Holder(obj))
    # i.e. a list of sub-estimators, each holding a framework model
    assert _holds_deep_learning_state([_Holder(obj)])


def test_holds_deep_learning_state_ignores_ordinary_values():
    """Test that values without framework state are not treated as opaque."""
    assert not _holds_deep_learning_state("foo")
    assert not _holds_deep_learning_state(42)
    assert not _holds_deep_learning_state(np.zeros(3))
    assert not _holds_deep_learning_state({"values": [1, 2]})
    assert not _holds_deep_learning_state(_Holder([1, 2]))


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
    estimator.state = _ConditionallyUnpickleable()

    before = _snapshot_state(estimator)
    assert before["state"][0] == "hash"

    estimator.state.pickleable = False

    assert _changed_state(before, estimator) == {"state"}
