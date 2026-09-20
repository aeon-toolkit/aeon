"""KGMTP transformer."""

__maintainer__ = ["johannfaouzi"]
__all__ = ["KGMTP"]

import multiprocessing
from itertools import combinations

import numpy as np
from numba import float32, get_num_threads, njit, prange, set_num_threads, vectorize
from scipy.signal import hilbert

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs

# Fixed kernel length used throughout the paper's kernel-grouping scheme: each of the 6
# positions in a length-6 kernel is either positively weighted or not, and kernels are
# grouped by how many positions are positive (1 through 5 -- a kernel with 0 or 6
# positive positions carries no discriminative information, so those two groups don't
# appear).
_KERNEL_LENGTH = 6

# Per-group multiplier applied to the weight matrix, for value_length 1 through 5
# (value_length 1-3 are left at x1). Fixed, paper-specific design choice; not a general
# formula in `_KERNEL_LENGTH`, so it's only meaningful for `_KERNEL_LENGTH == 6`. Also
# drives the "alpha" multiplier in `_build_kernel_group_tables` below.
_GROUP_MULTIPLIERS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 5.0}


def _build_kernel_group_tables():
    """Precompute the fixed per-kernel tables `_fit_biases`/`_transform` need.

    Returns
    -------
    indices : int32[:, :], shape (num_kernels, _KERNEL_LENGTH - 1)
    value_lengths, alpha_counts, gamma_counts : int32[:], shape (num_kernels,)
    group_sizes : int32[:], shape (_KERNEL_LENGTH - 1,)
    """
    kernel_length = _KERNEL_LENGTH
    max_value_length = kernel_length - 1

    rows, value_lengths, alpha_counts, gamma_counts, group_sizes = [], [], [], [], []
    for value_length in range(1, kernel_length):
        combos = list(combinations(range(kernel_length), value_length))
        group_sizes.append(len(combos))

        alpha = int(_GROUP_MULTIPLIERS[value_length])
        assert (alpha * kernel_length) % value_length == 0
        gamma = alpha * kernel_length // value_length

        for combo in combos:
            rows.append(list(combo) + [0] * (max_value_length - value_length))
            value_lengths.append(value_length)
            alpha_counts.append(alpha)
            gamma_counts.append(gamma)

    return (
        np.array(rows, dtype=np.int32),
        np.array(value_lengths, dtype=np.int32),
        np.array(alpha_counts, dtype=np.int32),
        np.array(gamma_counts, dtype=np.int32),
        np.array(group_sizes, dtype=np.int32),
    )


(
    _INDICES,
    _VALUE_LENGTHS,
    _ALPHA_COUNTS,
    _GAMMA_COUNTS,
    _GROUP_SIZES,
) = _build_kernel_group_tables()


# Numba kernels behind `_KGMTPBranch.fit`/`.transform`. Each kernel's "alpha"/"gamma"
# weighted sums are computed with a single multiplication instead of repeated addition
# -- mathematically equivalent, though not always bit-identical, due to floating-point
# non-associativity.


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    """Proportion-of-positive-values indicator: 1 if a > b else 0."""
    if a > b:
        return 1
    else:
        return 0


@njit(fastmath=True, cache=True, inline="always")
def _accumulate_alpha_gamma(A, G, C_alpha, C_gamma, dilation, padding, kernel_length):
    start = dilation
    end = len(A) - padding
    for gamma_index in range(kernel_length // 2):
        C_alpha[-end:] = C_alpha[-end:] + A[:end]
        C_gamma[gamma_index, -end:] = G[:end]
        end += dilation
    for gamma_index in range(kernel_length // 2 + 1, kernel_length):
        C_alpha[:-start] = C_alpha[:-start] + A[start:]
        C_gamma[gamma_index, :-start] = G[start:]
        start += dilation


# Single-threaded on purpose: the random example draws below are sequential and
# order-dependent (see `KGMTP`'s `random_state` parameter), unlike the
# embarrassingly-parallel `_transform` further down.
@njit(fastmath=True, parallel=False, cache=True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles, weights, rng):
    num_examples = X.shape[0]
    kernel_length = _KERNEL_LENGTH
    num_kernels = weights.shape[0]
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0
    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        input_length = X.shape[1] + dilation
        A = np.zeros(input_length, dtype=np.float64)
        G = np.zeros(input_length, dtype=np.float64)
        padding = (kernel_length * dilation) // 2
        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):
            value_length = _VALUE_LENGTHS[kernel_index]
            alpha = _ALPHA_COUNTS[kernel_index]
            gamma = _GAMMA_COUNTS[kernel_index]

            feature_index_end = feature_index_start + num_features_this_dilation
            _X1 = X[rng.integers(0, num_examples)]

            A[:-dilation] = -alpha * _X1
            G[:-dilation] = gamma * _X1

            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A
            C_gamma = np.zeros((kernel_length, input_length), dtype=np.float64)
            C_gamma[kernel_length // 2] = G
            _accumulate_alpha_gamma(
                A, G, C_alpha, C_gamma, dilation, padding, kernel_length
            )

            i0, i1, i2, i3, i4 = (
                _INDICES[kernel_index, 0],
                _INDICES[kernel_index, 1],
                _INDICES[kernel_index, 2],
                _INDICES[kernel_index, 3],
                _INDICES[kernel_index, 4],
            )
            if value_length == 1:
                C1 = C_alpha + C_gamma[i0]
            elif value_length == 2:
                C1 = C_alpha + C_gamma[i0] + C_gamma[i1]
            elif value_length == 3:
                C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
            elif value_length == 4:
                C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] + C_gamma[i3]
            else:
                C1 = (
                    C_alpha
                    + C_gamma[i0]
                    + C_gamma[i1]
                    + C_gamma[i2]
                    + C_gamma[i3]
                    + C_gamma[i4]
                )
            C1 = C1.astype(np.float32)

            biases[feature_index_start:feature_index_end] = np.quantile(
                C1, quantiles[feature_index_start:feature_index_end]
            )
            feature_index_start = feature_index_end

    return biases


@njit(
    "(float64[:,:],Tuple((int32[:],int32[:],float32[:],float32[:,:])),int32)",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform(
    X, parameters, n_features_per_kernel=5
) -> tuple[float32[:, :], float32[:, :]]:
    dilations, num_features_per_dilation, biases, weights = parameters
    kernel_length = weights.shape[1]

    num_examples = X.shape[0]
    num_kernels = len(weights)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    features = np.zeros(
        (num_examples, num_features * n_features_per_kernel), dtype=np.float32
    )
    features_hydra = np.zeros(
        (num_examples, num_kernels * num_dilations * 2), dtype=np.float32
    )

    for example_index in prange(num_examples):
        hydra_feature_index = 0
        feature_index_start = 0
        _X = X[example_index]
        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            input_length = _X.shape[0] + dilation
            padding = (kernel_length * dilation) // 2
            output_length = (
                _X.shape[0] + (2 * padding) - ((kernel_length - 1) * dilation)
            )
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_hydra_max = np.zeros(num_kernels, dtype=np.float32)
            C_hydra_min = np.zeros(num_kernels, dtype=np.float32)

            kernel_offset = 0
            for group in range(kernel_length - 1):
                value_length = group + 1
                group_size = _GROUP_SIZES[group]
                alpha = _ALPHA_COUNTS[kernel_offset]
                gamma = _GAMMA_COUNTS[kernel_offset]

                A = np.zeros(input_length, dtype=np.float64)
                G = np.zeros(input_length, dtype=np.float64)
                A[:-dilation] = -alpha * _X
                G[:-dilation] = gamma * _X

                C_alpha = np.zeros(input_length, dtype=np.float64)
                C_alpha[:] = A
                C_gamma = np.zeros((kernel_length, input_length), dtype=np.float64)
                C_gamma[kernel_length // 2, :] = G
                _accumulate_alpha_gamma(
                    A, G, C_alpha, C_gamma, dilation, padding, kernel_length
                )

                C_max = np.full(output_length, -np.inf)
                C_min = np.full(output_length, np.inf)
                C_max_index = np.zeros(output_length, dtype=np.int32)
                C_min_index = np.zeros(output_length, dtype=np.int32)

                for local_kernel_index in range(group_size):
                    kernel_index = kernel_offset + local_kernel_index
                    feature_index_end = feature_index_start + num_features_this_dilation

                    i0, i1, i2, i3, i4 = (
                        _INDICES[kernel_index, 0],
                        _INDICES[kernel_index, 1],
                        _INDICES[kernel_index, 2],
                        _INDICES[kernel_index, 3],
                        _INDICES[kernel_index, 4],
                    )
                    if value_length == 1:
                        C1 = C_alpha + C_gamma[i0]
                    elif value_length == 2:
                        C1 = C_alpha + C_gamma[i0] + C_gamma[i1]
                    elif value_length == 3:
                        C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
                    elif value_length == 4:
                        C1 = (
                            C_alpha
                            + C_gamma[i0]
                            + C_gamma[i1]
                            + C_gamma[i2]
                            + C_gamma[i3]
                        )
                    else:
                        C1 = (
                            C_alpha
                            + C_gamma[i0]
                            + C_gamma[i1]
                            + C_gamma[i2]
                            + C_gamma[i3]
                            + C_gamma[i4]
                        )
                    C = C1.astype(np.float32)

                    for j in range(C.shape[0]):
                        if C_max[j] <= C[j]:
                            C_max[j] = C[j]
                            C_max_index[j] = kernel_index
                        if C_min[j] >= C[j]:
                            C_min[j] = C[j]
                            C_min_index[j] = kernel_index

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0
                        zero_count = 0
                        for j in range(C.shape[0]):
                            if j < C.shape[0] - 1 and (
                                (C[j] > _bias and C[j + 1] < _bias)
                                or (C[j] < _bias and C[j + 1] > _bias)
                            ):
                                zero_count += 1

                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val

                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )
                        end = end + num_features
                        features[example_index, end] = zero_count / C.shape[0]
                    feature_index_start = feature_index_end

                for j in range(output_length):
                    C_hydra_max[C_max_index[j]] += C_max[j]
                    C_hydra_min[C_min_index[j]] += 1

                kernel_offset += group_size

            features_hydra[
                example_index, hydra_feature_index : hydra_feature_index + num_kernels
            ] = C_hydra_max
            hydra_feature_index += num_kernels
            features_hydra[
                example_index, hydra_feature_index : hydra_feature_index + num_kernels
            ] = C_hydra_min
            hydra_feature_index += num_kernels

    res = (features, features_hydra)
    return res


class _KGMTPBranch:
    """One kernel-grouping-and-pooling sub-transform."""

    def __init__(
        self, n_kernels=50_000, max_dilations_per_kernel=32, n_features_per_kernel=5
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel

    def fit(self, X, rng):
        weights = self._build_weights()
        num_kernels, kernel_length = weights.shape
        _, input_length = X.shape

        num_kernel_features = int(self.n_kernels / self.n_features_per_kernel)
        if num_kernel_features < num_kernels:
            raise ValueError(
                f"n_kernels // n_features_per_kernel must be at least "
                f"the number of kernels ({num_kernels}); got "
                f"{self.n_kernels} // {self.n_features_per_kernel} = "
                f"{num_kernel_features}. Increase `n_kernels`."
            )

        dilations, num_features_per_dilation = self._fit_dilations(
            input_length,
            num_kernel_features,
            self.max_dilations_per_kernel,
            num_kernels,
            kernel_length,
        )
        num_features_per_kernel = int(np.sum(num_features_per_dilation))
        quantiles = self._quantiles(num_kernels * num_features_per_kernel)

        biases = _fit_biases(
            X, dilations, num_features_per_dilation, quantiles, weights, rng
        )

        self.dilations_ = dilations
        self.num_features_per_dilation_ = num_features_per_dilation
        self.biases_ = biases
        self.weights_ = weights
        self.n_features_in_ = input_length
        return self

    def transform(self, X):
        parameters = (
            self.dilations_,
            self.num_features_per_dilation_,
            self.biases_,
            self.weights_,
        )
        features, features_hydra = _transform(X, parameters, self.n_features_per_kernel)
        features = np.nan_to_num(features)
        return features, features_hydra

    def _build_weights(self):
        length = _KERNEL_LENGTH
        blocks = []
        for value_length in range(1, length):
            combos = np.array(
                list(combinations(np.arange(length), value_length)), dtype=np.int32
            )
            num_kernels = len(combos)
            block = np.full((num_kernels, length), -1.0, dtype=np.float32)
            positive_value = (length - value_length) * 1 / value_length
            for i in range(num_kernels):
                for j in range(value_length):
                    block[i, combos[i, j]] = positive_value
            block *= _GROUP_MULTIPLIERS[value_length]
            blocks.append(block)
        return np.concatenate(blocks, axis=0)

    @staticmethod
    def _fit_dilations(
        input_length, num_features, max_dilations_per_kernel, num_kernels, kernel_length
    ):
        num_features_per_kernel = num_features // num_kernels
        true_max_dilations_per_kernel = min(
            num_features_per_kernel, max_dilations_per_kernel
        )
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((input_length - 1) / (kernel_length - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
                np.int32
            ),
            return_counts=True,
        )
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
            np.int32
        )

        remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)

        return dilations, num_features_per_dilation

    @staticmethod
    def _quantiles(n):
        return np.array(
            [(i * ((np.sqrt(5) + 1) / 2)) % 1 for i in range(1, n + 1)],
            dtype=np.float32,
        )


class KGMTP(BaseCollectionTransformer):
    """KG-MTP: kernel grouping with multiple transformations and pooling.

    Every series is fed through three representations (raw, its Hilbert transform, and
    its first difference), each fit and transformed by its own independent
    kernel-grouping-and-pooling sub-transform, with the resulting features concatenated
    across the three. Alongside the MiniRocket-style PPV-pooling features (extended to 5
    statistics per kernel: proportion of positive values, mean of positive values, mean
    index of positive values, longest below-bias stretch, and zero-crossing count), a
    second, Hydra-style block of per-kernel max/min-competition count features is also
    produced.

    If ``scale_hydra`` is ``True`` (the default), ``fit`` also fits a masked,
    epsilon-regularized scaler on the pooled Hydra features, and ``transform``
    applies it before concatenating the Hydra block with the raw PPV-pooling block.
    Set ``scale_hydra=False`` to get both blocks raw instead.

    Multivariate series are supported by processing each channel independently through
    this same raw/Hilbert/diff pipeline (with its own, separately fitted set of kernels
    per channel), then concatenating every channel's output into the final design
    matrix. Channels are never mixed, so ``n_kernels`` is the per-channel budget: the
    total feature budget scales with the number of channels.

    Parameters
    ----------
    n_kernels : int, default=50_000
        Total PPV-pooling feature budget per channel, split evenly across the three
        representations this transform computes internally (``n_kernels // 3`` each).
        For multivariate series, this same per-channel budget is used independently for
        every channel, so the total feature budget scales with the number of channels.
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per kernel.
    n_features_per_kernel : int, default=5
        Number of PPV-pooling statistics produced per kernel/dilation combination.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `transform`. ``-1`` means using all
        processors. Bias-fitting during `fit` is always single-threaded.
    scale_hydra : bool, default=True
        Whether to scale the pooled Hydra features with a masked, epsilon-regularized
        scaler fitted during `fit` (matching the original paper's pipeline). If
        ``False``, `transform`'s Hydra block is left raw, like the PPV-pooling block
        always is.
    random_state : int, ``numpy.random.Generator`` or None, default=None
        If ``int``, random_state is the seed used by the random number generator.
        If a ``Generator`` instance, random_state is the random number generator.
        If ``None``, use a fresh, unseeded ``Generator`` instance.
        Unlike most aeon estimators, a legacy ``numpy.random.RandomState`` is not
        accepted.

    Attributes
    ----------
    base_, hilbert_, diff_ : list of tuple, length n_channels
        Fitted kernel parameters for the raw, Hilbert-transform, and first-difference
        representations respectively, one entry per channel (in channel order), each
        entry a ``(dilations, num_features_per_dilation, biases, weights)`` tuple.
    n_ppv_features_ : int
        Width of the PPV-pooling block within `transform`'s output.
    n_hydra_features_ : int
        Width of the Hydra block within `transform`'s output.
    random_state_ : numpy.random.Generator
        The `Generator` instance used to fit this transform.

    See Also
    --------
    Rocket, MiniRocket, MultiRocket, HydraTransformer

    Notes
    -----
    Original code: https://github.com/WangPanJie2024/KG-MTP

    References
    ----------
    .. [1] Wang, Wu, Wei, Li. "KG-MTP: Kernel Grouping for Time Series Classification
           with Multiple Transformations and Pooling Operators." Expert Systems with
           Applications, 2025.
           https://doi.org/10.1016/j.eswa.2025.128693

    Examples
    --------
    >>> from aeon.transformations.collection.convolution_based import KGMTP
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> trf = KGMTP(n_kernels=1200)
    >>> trf.fit(X_train)
    KGMTP(n_kernels=1200)
    >>> X_train_t = trf.transform(X_train)
    >>> X_test_t = trf.transform(X_test)
    """

    _tags = {
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "capability:multithreading": True,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_kernels=50_000,
        max_dilations_per_kernel=32,
        n_features_per_kernel=5,
        scale_hydra=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.scale_hydra = scale_hydra
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y=None):
        """Fit the three branches and the Hydra scaler, for every channel.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training time series (equal length). Each channel is fit independently
            (see class docstring).
        y : ignored

        Returns
        -------
        self
        """
        self._fit_transform(X, y)
        return self

    def _fit_transform(self, X, y=None):
        """Fit the three branches per channel and the Hydra scaler; transform `X`.

        For each channel, fits the raw, Hilbert-transform, and first-difference
        branches in turn (each branch's `fit` already computes its own transform of
        that channel as a side effect, to determine dilations/biases). Channels'
        blocks are concatenated (see class docstring), then optionally fits the
        Hydra-count scaler on the pooled, multi-channel Hydra features and applies it
        if `scale_hydra=True`. The PPV-pooling features are left unscaled.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training time series (equal length). Each channel is fit independently
            (see class docstring).
        y : ignored

        Returns
        -------
        Xt : ndarray of shape (n_cases, n_ppv_features_ + n_hydra_features_)
            Raw (unscaled) PPV-pooling features followed by Hydra max/min-count
            features, scaled if `scale_hydra` is True (the default) and raw
            otherwise (see `n_ppv_features_`, `n_hydra_features_`).
        """
        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = self._check_random_state(self.random_state)
        self.random_state_ = rng

        n_kernels_per_branch = self.n_kernels // 3
        self._base, self._hilbert, self._diff = [], [], []
        self.base_, self.hilbert_, self.diff_ = [], [], []
        X_list, X_hilbert_list, X_diff_list = [], [], []

        for c in range(X.shape[1]):
            Xc = X[:, c, :].astype(np.float64)
            Xc_hilbert = self._hilbert_transform(Xc)
            Xc_diff = np.diff(Xc, 1)
            X_list.append(Xc)
            X_hilbert_list.append(Xc_hilbert)
            X_diff_list.append(Xc_diff)

            base = _KGMTPBranch(
                n_kernels_per_branch,
                self.max_dilations_per_kernel,
                self.n_features_per_kernel,
            ).fit(Xc, rng)
            hilbert = _KGMTPBranch(
                n_kernels_per_branch,
                self.max_dilations_per_kernel,
                self.n_features_per_kernel,
            ).fit(Xc_hilbert, rng)
            diff = _KGMTPBranch(
                n_kernels_per_branch,
                self.max_dilations_per_kernel,
                self.n_features_per_kernel,
            ).fit(Xc_diff, rng)

            self._base.append(base)
            self._hilbert.append(hilbert)
            self._diff.append(diff)
            self.base_.append(
                (
                    base.dilations_,
                    base.num_features_per_dilation_,
                    base.biases_,
                    base.weights_,
                )
            )
            self.hilbert_.append(
                (
                    hilbert.dilations_,
                    hilbert.num_features_per_dilation_,
                    hilbert.biases_,
                    hilbert.weights_,
                )
            )
            self.diff_.append(
                (
                    diff.dilations_,
                    diff.num_features_per_dilation_,
                    diff.biases_,
                    diff.weights_,
                )
            )

        train_features, train_hydra = self._transform_branches(
            X_list, X_hilbert_list, X_diff_list
        )
        self.n_ppv_features_ = train_features.shape[1]
        self.n_hydra_features_ = train_hydra.shape[1]

        if self.scale_hydra:
            self._hydra_mu_, self._hydra_sigma_ = self._sparse_scaler_fit(train_hydra)
            train_hydra = self._sparse_scaler_transform(
                train_hydra, self._hydra_mu_, self._hydra_sigma_
            )
        return np.concatenate([train_features, train_hydra], axis=1)

    def _transform(self, X, y=None):
        """Apply the fitted branches to every channel of `X`, scale, and concatenate.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Must have the same number of channels and series length `X` was fitted
            on.
        y : ignored

        Returns
        -------
        Xt : ndarray of shape (n_cases, n_ppv_features_ + n_hydra_features_)
            Raw (unscaled) PPV-pooling features followed by Hydra max/min-count
            features, scaled if `scale_hydra` is True (the default) and raw
            otherwise (see `n_ppv_features_`, `n_hydra_features_`).
        """
        X_list, X_hilbert_list, X_diff_list = [], [], []
        for c in range(X.shape[1]):
            Xc = X[:, c, :].astype(np.float64)
            X_list.append(Xc)
            X_hilbert_list.append(self._hilbert_transform(Xc))
            X_diff_list.append(np.diff(Xc, 1))

        features, hydra = self._transform_branches(X_list, X_hilbert_list, X_diff_list)
        if self.scale_hydra:
            hydra = self._sparse_scaler_transform(
                hydra, self._hydra_mu_, self._hydra_sigma_
            )
        return np.concatenate([features, hydra], axis=1)

    def _transform_branches(self, X_list, X_hilbert_list, X_diff_list):
        """Run each channel's fitted branches and concatenate every block."""
        prev_threads = get_num_threads()
        n_jobs = (
            multiprocessing.cpu_count()
            if self._n_jobs < 1 or self._n_jobs > multiprocessing.cpu_count()
            else self._n_jobs
        )
        set_num_threads(n_jobs)
        try:
            features_blocks, hydra_blocks = [], []
            for c, (Xc, Xc_hilbert, Xc_diff) in enumerate(
                zip(X_list, X_hilbert_list, X_diff_list)
            ):
                base_features, base_hydra = self._base[c].transform(Xc)
                hilbert_features, hilbert_hydra = self._hilbert[c].transform(Xc_hilbert)
                diff_features, diff_hydra = self._diff[c].transform(Xc_diff)

                features_blocks.append(
                    np.concatenate(
                        [base_features, hilbert_features, diff_features], axis=1
                    )
                )
                hydra_blocks.append(
                    np.concatenate([base_hydra, hilbert_hydra, diff_hydra], axis=1)
                )
        finally:
            set_num_threads(prev_threads)

        features = np.concatenate(features_blocks, axis=1)
        hydra = np.concatenate(hydra_blocks, axis=1)
        return features, hydra

    def _check_random_state(self, random_state):
        """Coerce `random_state` into a `numpy.random.Generator`."""
        if random_state is None:
            return np.random.default_rng()
        if isinstance(random_state, (int, np.integer)):
            return np.random.default_rng(random_state)
        if isinstance(random_state, np.random.Generator):
            return random_state
        raise TypeError(
            "random_state must be None, an int, or a numpy.random.Generator "
            f"(got {type(random_state).__name__}). A legacy "
            "numpy.random.RandomState is not supported here, since the "
            "Numba-jitted bias-fitting kernel only implements the modern "
            "Generator API in nopython mode."
        )

    @staticmethod
    def _hilbert_transform(X):
        """Vectorised discrete Hilbert transform."""
        out = -hilbert(X, axis=-1).imag
        return out.astype(np.float32).astype(np.float64)

    @staticmethod
    def _sparse_scaler_fit(X, exponent=4):
        """Fit a NumPy port of the original paper's (torch-based) `SparseScaler`.

        `torch.Tensor.std()` defaults to `unbiased=True` (`ddof=1`); NumPy's default is
        `ddof=0`, so `ddof=1` is passed explicitly below to match the original
        implementation.
        """
        X = np.sqrt(np.clip(X, 0, None))
        epsilon = (X == 0).mean(axis=0) ** exponent + 1e-8
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1) + epsilon
        return mu, sigma

    @staticmethod
    def _sparse_scaler_transform(X, mu, sigma):
        """Apply a fitted `_sparse_scaler_fit` scaler."""
        X = np.sqrt(np.clip(X, 0, None))
        return ((X - mu) * (X != 0)) / sigma

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        `n_kernels=1200` is the smallest budget that clears `_KGMTPBranch.fit`'s
        `n_kernels // n_features_per_kernel >= num_kernels(62)` floor once split three
        ways and divided by the default `n_features_per_kernel=5` (1200 // 3 // 5 = 80
        >= 62).
        """
        return {
            "n_kernels": 1200,
            "max_dilations_per_kernel": 4,
            "n_features_per_kernel": 5,
        }
