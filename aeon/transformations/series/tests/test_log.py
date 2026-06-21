"""Tests for LogTransformer."""

import numpy as np

from aeon.datasets import load_airline
from aeon.transformations.series._log import LogTransformer


def test_log_transform_against_numpy():
    """Test LogTransformer matches a direct numpy log computation."""
    y = load_airline()
    t = LogTransformer(offset=1.0, scale=2.0)
    Xt = t.fit_transform(y)

    expected = np.log(2.0 * (np.asarray(y) + 1.0))
    np.testing.assert_allclose(Xt.squeeze(), expected.squeeze())


def test_log_inverse_transform_roundtrip():
    """Test inverse_transform recovers the original series."""
    y = load_airline()
    t = LogTransformer(offset=1.0, scale=2.0)
    Xt = t.fit_transform(y)
    Xinv = t.inverse_transform(Xt)

    np.testing.assert_allclose(
        Xinv.squeeze(), np.asarray(y).squeeze(), rtol=1e-10, atol=1e-10
    )
