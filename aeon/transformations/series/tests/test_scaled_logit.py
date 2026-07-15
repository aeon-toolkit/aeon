"""ScaledLogit transform unit tests."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

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
    """Test forward and inverse values for each bound configuration."""
    transformer = ScaledLogitSeriesTransformer(lower, upper)
    y_transformed = transformer.fit_transform(TEST_SERIES)
    assert_array_equal(y_transformed.squeeze(), output)
    y_inverted = transformer.inverse_transform(y_transformed)
    np.testing.assert_allclose(
        y_inverted.squeeze(), TEST_SERIES, rtol=1e-10, atol=1e-10
    )


def test_scaled_logit_bound_warnings():
    """Tests all warnings."""
    with pytest.warns(RuntimeWarning, match="not have values lower than lower_bound"):
        ScaledLogitSeriesTransformer(lower_bound=300, upper_bound=0).fit_transform(
            TEST_SERIES
        )

    with pytest.warns(RuntimeWarning, match="not have values greater than upper_bound"):
        ScaledLogitSeriesTransformer(lower_bound=300, upper_bound=0).fit_transform(
            TEST_SERIES
        )
