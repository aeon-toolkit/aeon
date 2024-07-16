"""Tests for ExponentTransformer and SqrtTransformer."""

__maintainer__ = []
__all__ = []

import pytest

from aeon.testing.data_generation._legacy import make_series
from aeon.transformations.exponent import ExponentTransformer


@pytest.mark.parametrize("offset", ["a", [1, 2.3]])
def test_wrong_offset_type_raises_error(offset):
    """Test an error is raised for incorrect offset types."""
    y = make_series(n_timepoints=75)

    # Test input types
    match = f"Expected `offset` to be int or float, but found {type(offset)}."
    with pytest.raises(ValueError, match=match):
        transformer = ExponentTransformer(offset=offset)
        transformer.fit(y)


# Test only applies to PowerTransformer b/c SqrtTransformer doesn't have power
# hyperparameter
@pytest.mark.parametrize("power", ["a", [1, 2.3]])
def test_wrong_power_type_raises_error(power):
    """Test an error is raised for incorrect power types."""
    y = make_series(n_timepoints=75)

    # Test input types
    match = f"Expected `power` to be int or float, but found {type(power)}."
    with pytest.raises(ValueError, match=match):
        transformer = ExponentTransformer(power=power)
        transformer.fit(y)
