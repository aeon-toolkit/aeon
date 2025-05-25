"""Test ClaSP series transformer."""

import numpy as np

from aeon.transformations.series import ClaSPTransformer


def test_clasp():
    """Test ClaSP series transformer returned size."""
    for dtype in [np.float64, np.float32, np.float16]:
        series = np.arange(100, dtype=dtype)
        clasp = ClaSPTransformer()
        profile = clasp.fit_transform(series)

        m = len(series) - clasp.window_length + 1
        assert np.float64 == profile.dtype
        assert m == len(profile)
