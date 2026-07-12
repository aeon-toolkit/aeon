"""Equivalence tests for the optimised Catch22 transform paths.

Catch22 has several fast paths that must agree with each other and with the
per-series reference: a numba dispatch loop and batched autocorrelation/Welch
FFT caches for equal-length 3D input, a per-series fallback for unequal-length
input, and a feature skip-mask used for efficient predictions. These tests
check those paths produce the same features, rather than pinning exact values.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.feature_based._catch22 import (
    _compute_autocorrelations,
    feature_names,
    feature_names_short,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies

N_CASES = 8
N_CHANNELS = 3
N_TIMEPOINTS = 128
# Lengths spanning the autocorrelation FFT sizes: a power of two, one just above
# it (2**6 + 1 forces the next padded FFT size), and a short interval length.
AC_LENGTHS = [40, 65, 128]

# Features that aeon's numba code computes identically (to floating-point error)
# to the reference C implementation in pycatch22. The remaining features differ
# by small amounts: the histogram-mode, AMI and trev features round their
# binning arithmetic differently in C and numba, and the two outlier-timing
# features report values on a coarse quantised grid where the two implementations
# land on neighbouring steps. Those are not asserted against pycatch22 here.
PYCATCH22_EQUIVALENT_FEATURES = [
    "acf_timescale",
    "acf_first_min",
    "stretch_high",
    "transition_matrix",
    "periodicity",
    "ami_timescale",
    "whiten_timescale",
    "stretch_decreasing",
    "entropy_pairs",
    "rs_range",
    "dfa",
    "low_freq_power",
]


def _example(n_channels=1, n_timepoints=N_TIMEPOINTS, random_state=0):
    """Return an (N_CASES, n_channels, n_timepoints) example collection."""
    return make_example_3d_numpy(
        n_cases=N_CASES,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=random_state,
        return_y=False,
    )


@pytest.mark.parametrize("feature", feature_names_short)
def test_single_feature_matches_its_column_in_full_transform(feature):
    """Each feature computed alone equals its column in the full 22-feature output.

    Extracting a single feature builds only the caches that feature needs, while
    the full transform builds them all; the shared intermediates must give the
    same value either way.
    """
    X = _example()
    full = Catch22(features="all").fit_transform(X)
    single = Catch22(features=[feature]).fit_transform(X)

    column = feature_names_short.index(feature)
    assert_allclose(single[:, 0], full[:, column], atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("n_timepoints", AC_LENGTHS)
def test_multivariate_matches_independent_channels(n_timepoints):
    """A multivariate transform equals transforming each channel on its own.

    The batched caches flatten all channels into one FFT; the result for each
    channel block must match processing that channel in isolation.
    """
    X = _example(n_channels=N_CHANNELS, n_timepoints=n_timepoints)
    n_features = 22

    multivariate = Catch22(features="all").fit_transform(X)
    for channel in range(N_CHANNELS):
        single = Catch22(features="all").fit_transform(X[:, channel : channel + 1])
        block = multivariate[:, channel * n_features : (channel + 1) * n_features]
        assert_allclose(block, single, atol=1e-10, equal_nan=True)


def test_catch24_columns_are_series_mean_and_std():
    """catch24 appends the series mean and standard deviation as the last columns.

    These two columns are produced outside the numba kernel, so check they equal
    numpy's per-series mean and population standard deviation.
    """
    X = _example(n_channels=N_CHANNELS)
    n_features = 24

    Xt = Catch22(features="all", catch24=True).fit_transform(X)
    for channel in range(N_CHANNELS):
        mean_column = Xt[:, channel * n_features + 22]
        std_column = Xt[:, channel * n_features + 23]
        assert_allclose(mean_column, X[:, channel].mean(axis=1), atol=1e-10)
        assert_allclose(std_column, X[:, channel].std(axis=1), atol=1e-10)


def test_masked_features_are_zeroed_and_others_unchanged():
    """The skip-mask zeros masked features and leaves the rest equal to the full output.

    Interval forests set ``_transform_features`` to compute only the features a
    tree needs. Masking is per (channel, feature); with a multivariate input the
    batched caches must still return the correct value for every unmasked entry.
    """
    X = _example(n_channels=N_CHANNELS)
    n_features = 22

    reference = Catch22(features="all").fit_transform(X)

    # keep every second feature across the flattened (channel, feature) layout
    keep = np.ones(n_features * N_CHANNELS, dtype=bool)
    keep[1::2] = False

    masked = Catch22(features="all")
    masked.fit(X)
    masked._transform_features = keep.tolist()
    Xt = masked.transform(X)

    assert_allclose(Xt[:, keep], reference[:, keep], atol=1e-10, equal_nan=True)
    assert np.count_nonzero(Xt[:, ~keep]) == 0


def test_unequal_and_equal_length_paths_agree():
    """The per-series fallback agrees with the batched path for the same series.

    Unequal-length input takes the per-series path; the same series as a single
    equal-length collection takes the batched/kernel path. Both should produce
    the same features to within the FFT rounding difference between them.
    """
    rng = np.random.RandomState(0)
    lengths = [50, 65, 97]
    unequal_series = [rng.normal(size=(1, length)) for length in lengths]

    fallback = Catch22(features="all").fit_transform(unequal_series)
    for i, series in enumerate(unequal_series):
        batched = Catch22(features="all").fit_transform(series[np.newaxis])
        assert_allclose(fallback[i], batched[0], atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("n_timepoints", AC_LENGTHS)
def test_batched_autocorrelation_matches_per_series_reference(n_timepoints):
    """The batched autocorrelation cache matches the per-series reference.

    The cache computes every case/channel autocorrelation in one pocketfft call;
    it must match the per-series ``_compute_autocorrelations`` to FFT precision,
    and a constant (zero-variance) series must give an all-zero result.
    """
    X = _example(n_channels=N_CHANNELS, n_timepoints=n_timepoints)
    c22 = Catch22()

    # feature 2 (acf_timescale) is an autocorrelation feature, so requesting it
    # triggers the batched cache
    cache = c22._ac_batch_cache(X, [2], N_CASES)
    for i in range(N_CASES):
        for channel in range(N_CHANNELS):
            reference = _compute_autocorrelations(X[i, channel])
            assert_allclose(cache[i, channel], reference, atol=1e-12)

    constant = np.ones((N_CASES, N_CHANNELS, n_timepoints))
    constant_cache = c22._ac_batch_cache(constant, [2], N_CASES)
    assert np.count_nonzero(constant_cache) == 0


@pytest.mark.skipif(
    not _check_soft_dependencies("pycatch22", severity="none"),
    reason="skip test if required soft dependency pycatch22 not available",
)
def test_matches_pycatch22_reference_on_scale_invariant_features():
    """Aeon reproduces pycatch22's C output for its scale-invariant features.

    catch22 features are defined on z-normalised series, so both implementations
    are given the same z-normalised input (pycatch22 also z-normalises
    internally). For the features in PYCATCH22_EQUIVALENT_FEATURES aeon's numba
    code and the reference C library agree up to floating-point error.
    """
    import pycatch22

    X = _example(n_timepoints=200)
    z_normalised = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

    aeon = Catch22(features=PYCATCH22_EQUIVALENT_FEATURES).fit_transform(z_normalised)

    reference_names = pycatch22.catch22_all(list(range(30)))["names"]
    columns = [
        reference_names.index(feature_names[feature_names_short.index(feature)])
        for feature in PYCATCH22_EQUIVALENT_FEATURES
    ]
    reference = np.array(
        [pycatch22.catch22_all(X[i, 0].tolist())["values"] for i in range(N_CASES)]
    )[:, columns]

    assert_allclose(aeon, reference, atol=1e-6)
