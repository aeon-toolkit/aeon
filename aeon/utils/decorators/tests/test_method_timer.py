"""Tests for method_timer decorator."""

import pytest

from aeon.utils.decorators.method_timer import method_timer


def test_method_timer_sets_attr_and_returns_value():
    """Test sets the attribute and returns the method output."""

    class _A:
        @method_timer("t_ms")
        def f(self, x):
            """Docstring here."""
            return x + 1

    a = _A()
    out = a.f(3)

    assert out == 4
    assert hasattr(a, "t_ms")
    assert isinstance(a.t_ms, int)
    assert a.t_ms >= 0
    assert _A.f.__name__ == "f"
    assert _A.f.__doc__ == "Docstring here."


def test_method_timer_sets_attr_even_if_method_raises():
    """Test sets the attribute even if it raises an exception."""

    class _A:
        @method_timer("t_ms")
        def boom(self):
            raise ValueError("nope")

    a = _A()
    with pytest.raises(ValueError, match="nope"):
        a.boom()

    assert hasattr(a, "t_ms")
    assert isinstance(a.t_ms, int)
    assert a.t_ms >= 0


def test_method_timer_missing_self_raises_typeerror():
    """Test raises TypeError if the decorated function is missing self."""

    @method_timer("t_ms")
    def f(self):
        return 1

    with pytest.raises(
        TypeError, match=r"method_timer expects an instance method \(missing self\)\."
    ):
        f()


def test_method_timer_first_arg_not_instance_raises_typeerror():
    """Test raises TypeError if the first argument is not an instance."""

    @method_timer("t_ms")
    def f(self):
        return 1

    with pytest.raises(
        TypeError,
        match=r"method_timer expects first arg to be an instance; can't "
        r"set 't_ms' on int\.",
    ):
        f(123)
