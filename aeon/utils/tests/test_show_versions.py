"""Test the show versions function."""

from aeon.testing.utils.output_suppression import suppress_output
from aeon.utils.show_versions import show_versions


@suppress_output()
def test_show_versions():
    """Test show versions function."""
    show_versions()

    s = show_versions(as_str=True)
    assert isinstance(s, str)
    assert "System" in s
    assert "Python dependencies" in s
