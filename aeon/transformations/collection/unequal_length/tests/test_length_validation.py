"""Test unequal length transformer target length validation."""

import numpy as np
import pytest

from aeon.transformations.collection.unequal_length import Padder, Resizer, Truncator


@pytest.mark.parametrize(
    ("transformer", "parameter_name"),
    [
        (Padder, "padded_length"),
        (Resizer, "resized_length"),
        (Truncator, "truncated_length"),
    ],
)
@pytest.mark.parametrize("invalid_length", [False, True, 0, -1])
def test_integer_length_parameters_must_be_positive_non_bool(
    transformer, parameter_name, invalid_length
):
    """Test integer target length parameters reject bool and non-positive values."""
    with pytest.raises(
        ValueError,
        match=f"{parameter_name} must be a positive integer",
    ):
        transformer(**{parameter_name: invalid_length})


@pytest.mark.parametrize(
    ("transformer", "parameter_name", "valid_length"),
    [
        (Padder, "padded_length", 1),
        (Padder, "padded_length", np.int64(1)),
        (Padder, "padded_length", "min"),
        (Padder, "padded_length", "max"),
        (Resizer, "resized_length", 1),
        (Resizer, "resized_length", np.int64(1)),
        (Resizer, "resized_length", "min"),
        (Resizer, "resized_length", "max"),
        (Truncator, "truncated_length", 1),
        (Truncator, "truncated_length", np.int64(1)),
        (Truncator, "truncated_length", "min"),
        (Truncator, "truncated_length", "max"),
    ],
)
def test_valid_length_parameters(transformer, parameter_name, valid_length):
    """Test valid target length parameters are accepted."""
    transformer(**{parameter_name: valid_length})


def test_padder_rejects_array_fill_value():
    """Padder should reject an array-like fill_value with a clear error (gh-3495)."""
    X = [np.zeros((1, 3)), np.zeros((1, 5))]
    padder = Padder(padded_length=6, fill_value=np.array([1, 2]))
    with pytest.raises(ValueError, match="fill_value must be a scalar"):
        padder.fit_transform(X)
