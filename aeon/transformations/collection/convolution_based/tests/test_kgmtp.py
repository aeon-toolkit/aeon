"""KGMTP transformer tests."""

import numpy as np
import pytest

from aeon.transformations.collection.convolution_based._kgmtp import (
    KGMTP,
    _build_kernel_group_tables,
    _KGMTPBranch,
)


def test_kgmtp_n_kernels_too_small():
    """Test KGMTP raises a clear error when n_kernels is too small.

    `_KGMTPBranch.fit` needs the per-branch kernel-feature budget
    (`n_kernels // 3 // 5`) to be at least the fixed number of kernels (62).
    Below that, it would otherwise divide by zero downstream, so this is turned into an
    explicit `ValueError` instead.
    """
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    kgmtp = KGMTP(n_kernels=30)  # 30 // 3 // 5 = 2 < 62
    with pytest.raises(ValueError, match="n_kernels // 5"):
        kgmtp.fit(X)


def test_kgmtp_random_state_shared_randomstate_advances():
    """A shared `RandomState` advances across fit() calls rather than repeating."""
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    rs = np.random.RandomState(42)
    params = KGMTP._get_test_params()

    a = KGMTP(random_state=rs, **params).fit(X)
    b = KGMTP(random_state=rs, **params).fit(X)

    # base_ is a list (one entry per channel) of (dilations,
    # num_features_per_dilation, biases, weights); X here is univariate.
    assert not np.array_equal(a.base_[0][2], b.base_[0][2])


def test_kgmtp_random_state_int_reproducible():
    """The same int seed reproduces the same fitted state across instances."""
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    params = KGMTP._get_test_params()

    a = KGMTP(random_state=7, **params).fit(X)
    b = KGMTP(random_state=7, **params).fit(X)

    np.testing.assert_array_equal(a.base_[0][2], b.base_[0][2])


def test_kgmtp_random_state_rejects_generator():
    """A raw `numpy.random.Generator` is rejected, unlike `RandomState`."""
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    kgmtp = KGMTP(random_state=np.random.default_rng(0), **KGMTP._get_test_params())
    with pytest.raises(ValueError, match="cannot be used to seed"):
        kgmtp.fit(X)


def test_kgmtp_minimum_length():
    """Series of length <= 6 (the fixed kernel length) raise a clear ValueError."""
    for length in (3, 6):
        X = np.random.default_rng(0).random(size=(10, 1, length))
        with pytest.raises(ValueError, match="n_timepoints must be > 6"):
            KGMTP(random_state=0).fit(X)

    X_ok = np.random.default_rng(0).random(size=(10, 1, 7))
    KGMTP(random_state=0).fit(X_ok)  # must not raise


def test_kgmtp_feature_block_sizes():
    """`n_pooling_features_`/`n_hydra_features_` match `transform`'s output width.

    `transform` concatenates the raw pooling block and the (by default,
    scaled) Hydra count block into one array; these two fitted attributes
    are what let a caller split them back apart (see the class docstring).
    """
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    kgmtp = KGMTP(random_state=0, **KGMTP._get_test_params()).fit(X)
    Xt = kgmtp.transform(X)

    assert Xt.shape[1] == kgmtp.n_pooling_features_ + kgmtp.n_hydra_features_
    assert kgmtp.n_hydra_features_ > 0
    assert kgmtp.n_pooling_features_ > 0


def test_kgmtp_scale_hydra_false_leaves_hydra_raw():
    """`scale_hydra=False` skips the Hydra scaler entirely."""
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    params = KGMTP._get_test_params()

    kgmtp = KGMTP(random_state=0, scale_hydra=False, **params)
    Xt = kgmtp.fit_transform(X)

    assert not hasattr(kgmtp, "_hydra_mu_")
    assert not hasattr(kgmtp, "_hydra_sigma_")

    X2d = X[:, 0, :].astype(np.float64)
    X_hilbert = kgmtp._hilbert_transform(X2d)
    X_diff = np.diff(X2d, 1)
    raw_features, raw_hydra = kgmtp._transform_branches([X2d], [X_hilbert], [X_diff])

    np.testing.assert_array_equal(Xt[:, : kgmtp.n_pooling_features_], raw_features)
    np.testing.assert_array_equal(Xt[:, kgmtp.n_pooling_features_ :], raw_hydra)
    np.testing.assert_array_equal(Xt, kgmtp.transform(X))

    # scale_hydra=True (the default) does scale the Hydra block, and differs
    # from the raw one above.
    kgmtp_scaled = KGMTP(random_state=0, **params).fit(X)
    Xt_scaled = kgmtp_scaled.transform(X)
    assert not np.array_equal(
        Xt_scaled[:, kgmtp_scaled.n_pooling_features_ :], raw_hydra
    )


def test_kgmtp_multivariate_concatenates_channels_sequentially():
    """Multivariate series: each channel gets its own, separately fitted kernels."""
    n_channels = 3
    X = np.random.default_rng(1).random(size=(10, n_channels, 100))
    params = KGMTP._get_test_params()

    multi = KGMTP(random_state=0, scale_hydra=False, **params).fit(X)
    Xt_multi = multi.transform(X)

    assert len(multi.base_) == n_channels
    assert len(multi.hilbert_) == n_channels
    assert len(multi.diff_) == n_channels

    single = KGMTP(random_state=0, scale_hydra=False, **params).fit(X[:, :1, :])
    Xt_single = single.transform(X[:, :1, :])
    assert multi.n_pooling_features_ == n_channels * single.n_pooling_features_
    assert multi.n_hydra_features_ == n_channels * single.n_hydra_features_
    assert Xt_multi.shape[1] == n_channels * Xt_single.shape[1]

    for c in range(n_channels):
        Xc = X[:, c : c + 1, :]
        model_c = KGMTP(random_state=0, scale_hydra=False, **params).fit(Xc)
        assert model_c.n_pooling_features_ == single.n_pooling_features_
        assert model_c.n_hydra_features_ == single.n_hydra_features_


def test_kgmtp_n_jobs_does_not_change_output():
    """`n_jobs` must not change `_transform`'s numeric result."""
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


def test_sparse_scaler_fit_single_case_no_nan():
    """`_sparse_scaler_fit` stays finite when fit on a single row."""
    X = np.array([[1.0, 4.0, 0.0]])

    mu, sigma = KGMTP._sparse_scaler_fit(X)

    assert np.isfinite(mu).all()
    assert np.isfinite(sigma).all()
    assert (sigma > 0).all()

    Xs = np.sqrt(np.clip(X, 0, None))
    expected_epsilon = (Xs == 0).mean(axis=0) ** 4 + 1e-8
    np.testing.assert_allclose(mu, Xs[0])
    np.testing.assert_allclose(sigma, expected_epsilon)


def test_kgmtp_fit_transform_single_case_no_nan():
    """A single training case does not propagate NaNs through `fit_transform`."""
    X = np.random.default_rng(0).random(size=(1, 1, 100))
    kgmtp = KGMTP(random_state=0, **KGMTP._get_test_params())
    Xt = kgmtp.fit_transform(X)
    assert np.isfinite(Xt).all()


def test_kgmtp_respects_numba_thread_limit(monkeypatch):
    """`_transform_branches` caps `n_jobs` at Numba's configured thread limit."""
    from numba import config

    monkeypatch.setattr(config, "NUMBA_NUM_THREADS", 1)
    X = np.random.default_rng(0).random(size=(10, 1, 100))
    kgmtp = KGMTP(n_jobs=2, random_state=0, **KGMTP._get_test_params())
    Xt = kgmtp.fit_transform(X)  # must not raise
    assert np.isfinite(Xt).all()


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
    """`_check_random_state` accepts `None`, an int, or a `RandomState`."""
    kgmtp = KGMTP()

    assert isinstance(kgmtp._check_random_state(None), np.random.Generator)
    assert isinstance(kgmtp._check_random_state(0), np.random.Generator)
    assert isinstance(
        kgmtp._check_random_state(np.random.RandomState(0)), np.random.Generator
    )
