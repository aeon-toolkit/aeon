"""Tests that soft dependencies are handled correctly in modules."""

import re
from importlib import import_module

import pytest

from aeon.testing.testing_config import PR_TESTING
from aeon.testing.tests import ALL_AEON_MODULES, ALL_AEON_MODULES_NO_TESTS

modules = ALL_AEON_MODULES_NO_TESTS if PR_TESTING else ALL_AEON_MODULES


def test_module_crawl():
    """Test that we are crawling modules correctly."""
    assert "aeon.classification" in modules
    assert "aeon.classification.shapelet_based" in modules
    assert "aeon.classification.base" in modules
    assert "aeon.segmentation" in modules


@pytest.mark.parametrize("module", modules)
def test_module_soft_deps(module):
    """Test soft dependency imports in aeon modules.

    Imports all modules and catch exceptions due to missing dependencies.
    """
    try:
        import_module(module)
    except ModuleNotFoundError as e:  # pragma: no cover
        dependency = "unknown"
        match = re.search(r"\'(.+?)\'", str(e))
        if match:
            dependency = match.group(1)

        raise ModuleNotFoundError(
            f"The module: {module} should not require any soft dependencies, "
            f"but tried importing: '{dependency}'. Make sure soft dependencies are "
            f"properly isolated."
        ) from e
