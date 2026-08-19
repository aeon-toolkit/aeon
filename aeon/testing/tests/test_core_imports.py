"""Tests that non-core dependencies are handled correctly in modules."""

import re
from importlib import import_module

from aeon.testing.tests import ALL_AEON_MODULES_NO_TESTS

if __name__ == "__main__":
    """Test imports in aeon modules with core dependencies only.

    Imports all modules and catch exceptions due to missing dependencies.
    """
    for module in ALL_AEON_MODULES_NO_TESTS:
        try:
            import_module(module)
        except ModuleNotFoundError as e:  # pragma: no cover
            dependency = "unknown"
            match = re.search(r"\'(.+?)\'", str(e))
            if match:
                dependency = match.group(1)

            raise ModuleNotFoundError(
                f"The module: {module} should not require any non-core dependencies, "
                f"but tried importing: '{dependency}'. Make sure non-core dependencies "
                f"are properly isolated outside of tests/ directories."
            ) from e
