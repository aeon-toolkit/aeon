"""ScaledLogit transform unit tests."""

__maintainer__ = []

from warnings import warn

import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.transformations.series._scaled_logit import ScaledLogitSeriesTransformer

TEST_SERIES = np.array([30, 40, 60])


@pytest.mark.parametrize(
    "lower, upper, output",
    [
        (10, 70, np.log((TEST_SERIES - 10) / (70 - TEST_SERIES))),
        (None, 70, -np.log(70 - TEST_SERIES)),
        (10, None, np.log(TEST_SERIES - 10)),
        (None, None, TEST_SERIES),
    ],
)
def test_scaled_logit_transform(lower, upper, output):
    """Test that we get the right output."""
    transformer = ScaledLogitSeriesTransformer(lower, upper)
    y_transformed = transformer.fit_transform(TEST_SERIES)
    assert np.all(output == y_transformed)


@pytest.mark.parametrize(
    "lower, upper, message",
    [
        (
            0,
            300,
            (
                "X in ScaledLogitSeriesTransformer should not have values greater"
                "than upper_bound"
            ),
        ),
        (
            300,
            700,
            "X in ScaledLogitSeriesTransformer should not have values lower than "
            "lower_bound",
        ),
    ],
)
def test_scaled_logit_bound_errors(lower, upper, message):
    """Tests all exceptions."""
    y = load_airline()
    with pytest.warns(RuntimeWarning):
        ScaledLogitSeriesTransformer(lower, upper).fit_transform(y)
        warn(message, RuntimeWarning)
