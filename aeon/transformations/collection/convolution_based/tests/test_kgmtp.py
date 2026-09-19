"""KGMTP transformer tests."""

import numpy as np
import pytest

from aeon.transformations.collection.convolution_based._kgmtp import (
    _PPV,
    KGMTP,
    _build_kernel_group_tables,
    _KGMTPBranch,
)


def test_kgmtp_n_kernels_too_small():
    """Test KGMTP raises a clear error when n_kernels is too small.

    `_KGMTPBranch.fit` needs the per-branch kernel-feature budget
    (`n_kernels // 3 // n_features_per_kernel`) to be at least the fixed
    number of kernels (62); below that it would otherwise divide by zero
    downstream, so this is turned into an explicit `ValueError` instead.
    """
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    kgmtp = KGMTP(n_kernels=30)  # 30 // 3 // 5 = 2 < 62
    with pytest.raises(ValueError, match="n_kernels // n_features_per_kernel"):
        kgmtp.fit(X)


def test_kgmtp_random_state_generator_advances():
    """A shared Generator advances across fit() calls rather than repeating.

    Matches how `KGMTP` is meant to be used across many resamples of a
    benchmark: one `Generator`, passed to several `KGMTP` instances,
    continuously advancing rather than reset per instance.
    """
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    rng = np.random.default_rng(42)
    params = KGMTP._get_test_params()

    a = KGMTP(random_state=rng, **params).fit(X)
    b = KGMTP(random_state=rng, **params).fit(X)

    # base_ is (dilations, num_features_per_dilation, biases, weights).
    assert not np.array_equal(a.base_[2], b.base_[2])


def test_kgmtp_random_state_rejects_legacy_randomstate():
    """A legacy `numpy.random.RandomState` is explicitly rejected.

    Unlike most aeon estimators, `random_state` here only accepts an int,
    `None`, or a `numpy.random.Generator` -- the bias-fitting kernel is
    Numba-compiled and only implements the modern `Generator` API in
    nopython mode.
    """
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    kgmtp = KGMTP(random_state=np.random.RandomState(0), **KGMTP._get_test_params())
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        kgmtp.fit(X)


def test_kgmtp_feature_block_sizes():
    """`n_ppv_features_`/`n_hydra_features_` match `transform`'s output width.

    `transform` concatenates the raw PPV-pooling block and the (by default,
    scaled) Hydra count block into one array; these two fitted attributes
    are what let a caller split them back apart (see the class docstring).
    """
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    kgmtp = KGMTP(random_state=0, **KGMTP._get_test_params()).fit(X)
    Xt = kgmtp.transform(X)

    assert Xt.shape[1] == kgmtp.n_ppv_features_ + kgmtp.n_hydra_features_
    assert kgmtp.n_hydra_features_ > 0
    assert kgmtp.n_ppv_features_ > 0


def test_kgmtp_scale_hydra_false_leaves_hydra_raw():
    """`scale_hydra=False` skips the Hydra scaler entirely.

    The Hydra block of `transform`'s output should then equal the raw,
    unscaled Hydra features `_transform_branches` computes directly, and no
    scaler should be fitted (`_hydra_mu_`/`_hydra_sigma_` unset). Also checks
    that `fit_transform` and `fit().transform()` agree with each other in
    this mode, since `_fit_transform` is an optimized override rather than
    the base class's default fit-then-transform.
    """
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    params = KGMTP._get_test_params()

    kgmtp = KGMTP(random_state=0, scale_hydra=False, **params)
    Xt = kgmtp.fit_transform(X)

    assert not hasattr(kgmtp, "_hydra_mu_")
    assert not hasattr(kgmtp, "_hydra_sigma_")

    X2d = X[:, 0, :].astype(np.float64)
    X_hilbert = kgmtp._hilbert_transform(X2d)
    X_diff = np.diff(X2d, 1)
    raw_features, raw_hydra = kgmtp._transform_branches(X2d, X_hilbert, X_diff)

    np.testing.assert_array_equal(Xt[:, : kgmtp.n_ppv_features_], raw_features)
    np.testing.assert_array_equal(Xt[:, kgmtp.n_ppv_features_ :], raw_hydra)
    np.testing.assert_array_equal(Xt, kgmtp.transform(X))

    # scale_hydra=True (the default) does scale the Hydra block, and differs
    # from the raw one above.
    kgmtp_scaled = KGMTP(random_state=0, **params).fit(X)
    Xt_scaled = kgmtp_scaled.transform(X)
    assert not np.array_equal(Xt_scaled[:, kgmtp_scaled.n_ppv_features_ :], raw_hydra)


def test_kgmtp_n_jobs_does_not_change_output():
    """`n_jobs` must not change `_transform`'s numeric result.

    It only changes how the per-example loop is threaded (via
    `numba.set_num_threads`).
    """
    X_train = np.random.default_rng(0).random(size=(10, 1, 100))
    X_test = np.random.default_rng(1).random(size=(6, 1, 100))
    params = KGMTP._get_test_params()

    kgmtp_1 = KGMTP(random_state=0, n_jobs=1, **params).fit(X_train)
    kgmtp_2 = KGMTP(random_state=0, n_jobs=2, **params).fit(X_train)

    np.testing.assert_array_equal(kgmtp_1.transform(X_test), kgmtp_2.transform(X_test))


def test_fit_dilations():
    """`_KGMTPBranch._fit_dilations` matches known values for a fixed case."""
    dilations, num_features_per_dilation = _KGMTPBranch._fit_dilations(
        input_length=100,
        num_features=1240,
        max_dilations_per_kernel=4,
        num_kernels=62,
        kernel_length=6,
    )
    assert np.array_equal(dilations, np.array([1, 2, 7, 19], dtype=np.int32))
    assert np.array_equal(
        num_features_per_dilation, np.array([5, 5, 5, 5], dtype=np.int32)
    )


def test_ppv():
    """`_PPV` is the proportion-of-positive-values indicator: `a > b`."""
    assert _PPV(np.float32(1), np.float32(0)) == 1
    assert _PPV(np.float32(0), np.float32(1)) == 0
    assert _PPV(np.float32(1), np.float32(1)) == 0


def test_build_kernel_group_tables_shapes_and_counts():
    """The kernel-grouping tables cover all 62 length-6 kernels once each.

    62 = sum(C(6, k) for k in 1..5) -- the same constant used as the
    `num_kernels` floor `_KGMTPBranch.fit` checks `n_kernels` against.
    """
    indices, value_lengths, alpha_counts, gamma_counts, group_sizes = (
        _build_kernel_group_tables()
    )
    assert np.array_equal(group_sizes, np.array([6, 15, 20, 15, 6], dtype=np.int32))
    assert group_sizes.sum() == 62
    assert indices.shape == (62, 5)
    assert value_lengths.shape == (62,)
    _, counts = np.unique(value_lengths, return_counts=True)
    assert np.array_equal(counts, np.array([6, 15, 20, 15, 6]))
    assert set(np.unique(alpha_counts).tolist()) == {1, 2, 5}
    assert set(np.unique(gamma_counts).tolist()) == {2, 3, 6}


def test_build_weights_rows_sum_to_zero():
    """Each kernel is mean-centered: positive/negative positions cancel out.

    `value_length * positive_value - (kernel_length - value_length) == 0`
    algebraically, before the group multiplier (which just rescales each
    row, so it stays zero).
    """
    weights = _KGMTPBranch()._build_weights()
    assert weights.shape == (62, 6)
    np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-4)


def test_quantiles():
    """`_KGMTPBranch._quantiles` is the golden-ratio low-discrepancy sequence."""
    q = _KGMTPBranch._quantiles(5)
    phi = (np.sqrt(5) + 1) / 2
    expected = np.array([(i * phi) % 1 for i in range(1, 6)], dtype=np.float32)
    assert q.shape == (5,)
    assert np.all((q >= 0) & (q < 1))
    np.testing.assert_allclose(q, expected)


def test_sparse_scaler_fit_transform():
    """`_sparse_scaler_fit`/`_transform` match a direct NumPy reimplementation.

    Uses a column of all zeros to also check that zero entries transform
    to exactly zero, via the `(X != 0)` mask in `_sparse_scaler_transform`.
    """
    X = np.array([[0.0, 4.0], [0.0, 9.0], [0.0, 16.0], [0.0, 25.0]])

    mu, sigma = KGMTP._sparse_scaler_fit(X)

    Xs = np.sqrt(np.clip(X, 0, None))
    epsilon = (Xs == 0).mean(axis=0) ** 4 + 1e-8
    expected_mu = Xs.mean(axis=0)
    expected_sigma = Xs.std(axis=0, ddof=1) + epsilon
    np.testing.assert_allclose(mu, expected_mu)
    np.testing.assert_allclose(sigma, expected_sigma)

    Xt = KGMTP._sparse_scaler_transform(X, mu, sigma)
    np.testing.assert_array_equal(Xt[:, 0], np.zeros(4))
    np.testing.assert_allclose(Xt[:, 1], (Xs[:, 1] - mu[1]) / sigma[1])


def test_check_random_state():
    """`_check_random_state` accepts `None`, an int, or an existing `Generator`."""
    kgmtp = KGMTP()

    assert isinstance(kgmtp._check_random_state(None), np.random.Generator)
    assert isinstance(kgmtp._check_random_state(0), np.random.Generator)

    rng = np.random.default_rng(0)
    assert kgmtp._check_random_state(rng) is rng
