"""Tests for repr utilities."""

from aeon.utils.repr import get_unchanged_and_required_params_as_str as get_str


class _Dummy:
    def __init__(self, a, b=1, c="x"):
        self.a, self.b, self.c = a, b, c


class _AllDefault:
    def __init__(self, b=1, c="x"):
        self.b, self.c = b, c


class _VarArgs:
    def __init__(self, a=1, *args, **kwargs):
        self.a = a


def test_required_param_always_included():
    """Parameters without a default are always reported."""
    assert get_str(_Dummy(a=5)) == "a=5"


def test_changed_default_included():
    """Parameters changed from their default are reported."""
    assert get_str(_Dummy(a=5, b=2)) == "a=5, b=2"


def test_strings_are_quoted():
    """String values are quoted in the output."""
    assert get_str(_Dummy(a="hi", c="y")) == "a='hi', c='y'"


def test_all_defaults_returns_empty_string():
    """No required or changed parameters gives an empty string."""
    assert get_str(_AllDefault()) == ""


def test_var_args_and_kwargs_ignored():
    """*args and **kwargs are not reported as required parameters."""
    assert get_str(_VarArgs()) == ""
    assert get_str(_VarArgs(a=2)) == "a=2"
