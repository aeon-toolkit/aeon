"""Tests for OptionalPassthrough transformer."""

import pytest
from pandas.testing import assert_series_equal

from aeon.testing.utils.data_gen import make_series
from aeon.transformations.boxcox import BoxCoxTransformer
from aeon.transformations.compose import OptionalPassthrough


@pytest.mark.parametrize("passthrough", [True, False])
def test_passthrough(passthrough):
    """Test that passthrough works as expected."""
    y = make_series(n_columns=1)

    optional_passthourgh = OptionalPassthrough(
        BoxCoxTransformer(), passthrough=passthrough
    )
    box_cox = BoxCoxTransformer()

    y_hat_passthrough = optional_passthourgh.fit_transform(y)
    y_inv_passthrough = optional_passthourgh.inverse_transform(y_hat_passthrough)

    y_hat_boxcox = box_cox.fit_transform(y)
    y_inv_boxcox = box_cox.inverse_transform(y_hat_boxcox)

    assert_series_equal(y, y_inv_passthrough)

    if passthrough:
        assert_series_equal(y, y_hat_passthrough)
        with pytest.raises(AssertionError):
            assert_series_equal(y_hat_boxcox, y_hat_passthrough)
    else:
        assert_series_equal(y_hat_passthrough, y_hat_boxcox)
        assert_series_equal(y_inv_passthrough, y_inv_boxcox)

        with pytest.raises(AssertionError):
            assert_series_equal(y, y_hat_passthrough)
