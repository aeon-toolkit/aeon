"""Tests for the PeriodogramTransformer."""

import numpy as np
import pytest

from aeon.transformations.collection import PeriodogramTransformer


@pytest.mark.parametrize(
    "pad_with",
    [
        "constant",
        "mean",
        "edge",
        "reflect",
        "median",
        "linear_ramp",
        "maximum",
        "minimum",
        "symmetric",
        "wrap",
    ],
)
def test_periodogram_unequal_length_pad_mode(pad_with):
    """Test all numpy.pad modes work on unequal length input and match numpy3D."""
    x0 = np.random.RandomState(0).rand(2, 20)
    x1 = np.random.RandomState(1).rand(2, 33)

    tnf = PeriodogramTransformer(pad_series=True, pad_with=pad_with)
    Xt_list = tnf.fit_transform([x0, x1])

    assert Xt_list[0].shape == (2, 16)
    assert Xt_list[1].shape == (2, 32)

    Xt_array = tnf.fit_transform(x0[np.newaxis])
    np.testing.assert_allclose(Xt_list[0], Xt_array[0])
