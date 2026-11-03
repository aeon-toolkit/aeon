"""MultiRocket regression tests."""

import numpy as np
import pytest

from aeon.transformations.collection.convolution_based import MultiRocket


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("n_timepoints", [8, 9])
def test_multirocket_rejects_short_series_after_differencing(n_channels, n_timepoints):
    """Reject series too short for the differenced representation."""
    transformer = MultiRocket(n_kernels=168)
    with pytest.raises(ValueError, match="n_timepoints must be >= 10"):
        transformer.fit(np.zeros((2, n_channels, n_timepoints)))


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("n_timepoints", [10, 11])
def test_multirocket_accepts_minimum_series(n_channels, n_timepoints):
    """Fit the minimum valid series lengths on both the uni- and multivariate path."""
    MultiRocket(n_kernels=168).fit(np.zeros((2, n_channels, n_timepoints)))
