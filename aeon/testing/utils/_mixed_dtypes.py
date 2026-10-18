"""Shared data and helpers for elastic distance tests on mixed dtype inputs."""

import warnings

import numpy as np

MIXED_DTYPES = [
    (np.float64, np.int64),
    (np.int64, np.float64),
    (np.float32, np.float64),
    (np.float64, np.float32),
]
SHORT_LENGTH = 9
LONG_LENGTH = 14
# Equal lengths, then both directions of the internal swap to the longer series.
MIXED_LENGTHS = [
    (LONG_LENGTH, LONG_LENGTH),
    (SHORT_LENGTH, LONG_LENGTH),
    (LONG_LENGTH, SHORT_LENGTH),
]
MIXED_CHANNELS = [1, 3]
N_CASES_X = 3
N_CASES_Y = 2
WINDOW = {"window": 0.5}
ITAKURA = {"itakura_max_slope": 0.6}
# Spread the values so that integer series are not mostly zeros.
SCALE = 5
RTOL_DOUBLE = 1e-12
RTOL_SINGLE = 1e-6


def _make_mixed(shape, dtype, rng):
    """Random array of the given shape and dtype."""
    return (rng.normal(size=shape) * SCALE).astype(dtype)


def _mixed_rtol(*dtypes):
    """Tolerance for comparing kernels, looser if any input is single precision."""
    return RTOL_SINGLE if np.float32 in dtypes else RTOL_DOUBLE


def _call_bounded(func, x, y, bounding, **kwargs):
    """Call a distance function, checking it warns only if bounding is requested."""
    import pytest

    if bounding:
        with pytest.warns(FutureWarning):
            return func(x, y, **bounding, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return func(x, y, **kwargs)
