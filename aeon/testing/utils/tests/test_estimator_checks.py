"""Tests for estimator checking utilities."""

from aeon.testing.utils.estimator_checks import _changed_state, _snapshot_state


class _DummyEstimator:
    """Minimal estimator implementing the tag interface used by the utilities."""

    def __init__(self, cant_pickle=False):
        self._cant_pickle = cant_pickle

    def get_tag(self, tag_name, raise_error=False, tag_value_default=None):
        if tag_name == "cant_pickle":
            return self._cant_pickle
        return tag_value_default


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


def test_snapshot_state_detects_nested_mutation():
    """Test that hashes detect mutation inside a top-level attribute."""
    estimator = _DummyEstimator()
    estimator.state = {"values": [1, 2]}

    before = _snapshot_state(estimator)

    assert before["state"][0] == "hash"
    assert _changed_state(before, vars(estimator)) == set()

    estimator.state["values"].append(3)

    assert _changed_state(before, vars(estimator)) == {"state"}


def test_snapshot_state_falls_back_to_identity():
    """Test identity fallback and replacement detection for unpickleable state."""
    estimator = _DummyEstimator()
    estimator.state = _Unpickleable()

    before = _snapshot_state(estimator)

    assert before["state"] == ("identity", estimator.state)
    assert _changed_state(before, vars(estimator)) == set()

    estimator.state = _Unpickleable()

    assert _changed_state(before, vars(estimator)) == {"state"}


def test_snapshot_state_respects_cant_pickle_tag():
    """Test that cant_pickle estimators use identity snapshots."""
    estimator = _DummyEstimator(cant_pickle=True)
    estimator.state = {"values": [1, 2]}

    before = _snapshot_state(estimator)

    assert before["state"] == ("identity", estimator.state)

    # In-place changes to opaque state cannot be detected by identity.
    estimator.state["values"].append(3)
    assert _changed_state(before, vars(estimator)) == set()

    estimator.state = {"values": [1, 2, 3]}
    assert _changed_state(before, vars(estimator)) == {"state"}


def test_changed_state_detects_added_and_removed_attributes():
    """Test that changes to the set of estimator attributes are detected."""
    estimator = _DummyEstimator()
    estimator.state = [1, 2, 3]

    before = _snapshot_state(estimator)

    del estimator.state
    estimator.added = "new"

    assert _changed_state(before, vars(estimator)) == {"added", "state"}


def test_changed_state_detects_value_becoming_unpickleable():
    """Test that failure to hash a previously hashable value counts as a change."""
    estimator = _DummyEstimator()
    estimator.state = _ConditionallyUnpickleable()

    before = _snapshot_state(estimator)
    assert before["state"][0] == "hash"

    estimator.state.pickleable = False

    assert _changed_state(before, vars(estimator)) == {"state"}
