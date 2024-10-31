"""Test functions in the _dependencies file."""

import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies


# Currently just partial testing of soft dep checks.
def test__check_soft_dependencies():
    """Test _check_soft_dependencies function."""
    with pytest.raises(TypeError, match="packages must be str or tuple of str"):
        _check_soft_dependencies((1, 2, 3))
    with pytest.raises(TypeError, match="package_import_alias must be a dict"):
        _check_soft_dependencies("tslearn", package_import_alias="alias")
    with pytest.raises(TypeError, match="must be a dict with str keys"):
        _check_soft_dependencies("tslearn", package_import_alias={1: "1", 2: "2"})
    with pytest.raises(TypeError, match="must be a dict with str keys and values"):
        _check_soft_dependencies("tslearn", package_import_alias={"1": 1, "2": 2})
    with pytest.raises(ModuleNotFoundError, match="No module named 'FOOBAR'"):
        _check_soft_dependencies("FOOBAR")
    with pytest.raises(ModuleNotFoundError, match="No module named 'FOOBAR'"):
        _check_soft_dependencies("FOOBAR", suppress_import_stdout=True)
    with pytest.raises(ModuleNotFoundError, match="FOOBAR requires package 'FOOBAR'"):
        _check_soft_dependencies("FOOBAR", obj="FOOBAR")
    with pytest.raises(RuntimeError, match="severity argument must be "):
        _check_soft_dependencies("FOOBAR", severity="FOOBAR")
    assert not _check_soft_dependencies("FOOBAR", severity="none")
    with pytest.warns(UserWarning, match="No module named 'FOOBAR'"):
        _check_soft_dependencies("FOOBAR", severity="warning")
