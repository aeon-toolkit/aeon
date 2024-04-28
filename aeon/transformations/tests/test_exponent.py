"""Tests for ExponentTransformer and SqrtTransformer."""

__maintainer__ = []
__all__ = []

import pytest

from aeon.testing.utils.data_gen import make_series
from aeon.transformations.exponent import ExponentTransformer, SqrtTransformer

power_transformers = [ExponentTransformer, SqrtTransformer]


@pytest.mark.parametrize("power_transformer", power_transformers)
@pytest.mark.parametrize("_offset", ["a", [1, 2.3]])
def test_wrong_offset_type_raises_error(power_transformer, _offset):
    """Test an error is raised for incorrect offset types."""
    y = make_series(n_timepoints=75)

    # Test input types
    match = f"Expected `offset` to be int or float, but found {type(_offset)}."
    with pytest.raises(ValueError, match=match):
        transformer = power_transformer(offset=_offset)
        transformer.fit(y)


# Test only applies to PowerTransformer b/c SqrtTransformer doesn't have power
# hyperparameter
@pytest.mark.parametrize("power_transformer", power_transformers[:1])
@pytest.mark.parametrize("_power", ["a", [1, 2.3]])
def test_wrong_power_type_raises_error(power_transformer, _power):
    """Test an error is raised for incorrect power types."""
    y = make_series(n_timepoints=75)

    # Test input types
    match = f"Expected `power` to be int or float, but found {type(_power)}."
    with pytest.raises(ValueError, match=match):
        transformer = power_transformer(power=_power)
        transformer.fit(y)
