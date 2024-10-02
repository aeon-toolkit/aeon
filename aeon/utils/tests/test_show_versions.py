"""Test the show versions function."""

from aeon.utils._show_versions import _show_versions


def test_show_versions():
    """Test show versions function."""
    str = _show_versions()
    assert "System" in str
    assert "Python dependencies" in str
