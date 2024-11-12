"""BORF transformer.

multivariate dictionary based transformer based on Bag-Of-Receptive-Fields transform.
"""

__maintainer__ = ["fspinna"]
__all__ = ["BORF"]

import itertools
import math
from collections.abc import Sequence
from typing import Literal, Optional

import numba as nb
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline as make_pipeline_sklearn

from aeon.transformations.collection import BaseCollectionTransformer


class BORF(BaseCollectionTransformer):
    """
    Bag-of-Receptive-Fields (BORF) Transformer.

    Transforms time series data into a bag-of-receptive-fields representation [1] using
    SAX. This transformer extracts SAX words from time series by applying sliding
    windows, discretizing, and counting the occurrences of the SAX words in the
    time series. The output is a sparse feature representation suitable for
    various downstream tasks such as classification and regression.

    Parameters
    ----------
    window_size_min_window_size : int, default=4
        The minimum window size for the sliding window.
    window_size_max_window_size : int or None, default=None
        The maximum window size for the sliding window. If None, it is set
        via a heuristic function based on the time series length.
    word_lengths_n_word_lengths : int, default=4
        The length of the words used in the SAX representation.
    alphabets_min_symbols : int, default=3
        The minimum number of symbols in the alphabet.
    alphabets_max_symbols : int, default=4
        The maximum number of symbols in the alphabet.
    alphabets_step : int, default=1
        The step size when iterating over the number of symbols in the alphabet.
    dilations_min_dilation : int, default=1
        The minimum dilation factor.
    dilations_max_dilation : int or None, default=None
        The maximum dilation factor. If None, it is set via a heuristic function
        based on the time series length.
    min_window_to_signal_std_ratio : float, default=0.0
        Minimum ratio of the window size to the standard deviation of the signal.
        If the window standard deviation is lower than this ratio,
        the window is considered "flat" and set to all zeros to avoid amplifying noise.
    n_jobs : int, default=1
        The number of `IndividualBORF` instances to run in parallel.
    n_jobs_numba : int, default=1
        The number of threads used for parallelizing each
        configuration inside each `IndividualBORF`.
    transformer_weights : array-like or None, default=None
        Weights applied to each transformer in the pipeline.
    complexity : {'quadratic', 'linear'}, default='quadratic'
        The computational complexity mode:
        - `'quadratic'`: Higher accuracy with more computational cost.
        - `'linear'`: Faster computations with potentially lower accuracy.
    densify : bool, default=False
        If True, converts the output to a dense array.
        By default, the output is returned as a scipy sparse matrix.


    Attributes
    ----------
    pipe_ : sklearn.pipeline.Pipeline
        The internal pipeline used for transforming the data.
    configs_ : dict
        Configuration parameters used in the pipeline.

    References
    ----------
    .. [1] F. Spinnato, R. Guidotti, A. Monreale and M. Nanni,
    "Fast, Interpretable and Deterministic Time Series Classification
    with a Bag-Of-Receptive-Fields," in IEEE Access,
    doi: 10.1109/ACCESS.2024.3464743

    Examples
    --------
    >>> from aeon.transformations.collection.dictionary_based import BORF
    >>> from aeon.datasets import load_unit_test
    >>> X, _ = load_unit_test()
    >>> borf = BORF()  # doctest: +SKIP
    >>> borf.fit(X)  # doctest: +SKIP
    BORF()
    >>> X_transformed = borf.transform(X)  # doctest: +SKIP

    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:inverse_transform": False,
        "capability:missing_values": True,
        "capability:multivariate": True,
        "capability:multithreading": True,
        "input_data_type": "Collection",
        "algorithm_type": "dictionary",
        "output_data_type": "Tabular",
        "requires_y": False,
        "python_dependencies": "sparse",
        "cant_pickle": True,
        "non_deterministic": True,
    }

    def __init__(
        self,
        window_size_min_window_size=4,
        window_size_max_window_size=None,
        word_lengths_n_word_lengths=4,
        alphabets_min_symbols=3,
        alphabets_max_symbols=4,
        alphabets_step=1,
        dilations_min_dilation=1,
        dilations_max_dilation=None,
        min_window_to_signal_std_ratio: float = 0.0,
        n_jobs=1,
        n_jobs_numba=1,
        transformer_weights=None,
        complexity: Literal["quadratic", "linear"] = "quadratic",
        densify=False,
    ):
        self.window_size_min_window_size = window_size_min_window_size
        self.window_size_max_window_size = window_size_max_window_size
        self.word_lengths_n_word_lengths = word_lengths_n_word_lengths
        self.alphabets_min_symbols = alphabets_min_symbols
        self.alphabets_max_symbols = alphabets_max_symbols
        self.alphabets_step = alphabets_step
        self.dilations_min_dilation = dilations_min_dilation
        self.dilations_max_dilation = dilations_max_dilation
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.n_jobs = n_jobs
        self.n_jobs_numba = n_jobs_numba
        self.transformer_weights = transformer_weights
        self.complexity = complexity
        self.densify = densify
        super().__init__()

    def _fit(self, X, y=None):
        time_series_length = X.shape[2]
        # for better computation time, this should be moved to the init,
        #  setting time_series_length as a user parameter

        pipeline_objects = [
            (_ReshapeTo2D, dict()),
            (_ZeroColumnsRemover, dict()),
            (_ToScipySparse, dict()),
        ]
        if self.densify:
            pipeline_objects.append((_ToDense, dict()))

        self.pipe_, self.configs_ = _build_pipeline_auto(
            time_series_min_length=time_series_length,
            time_series_max_length=time_series_length,
            window_size_min_window_size=self.window_size_min_window_size,
            window_size_max_window_size=self.window_size_max_window_size,
            word_lengths_n_word_lengths=self.word_lengths_n_word_lengths,
            alphabets_min_symbols=self.alphabets_min_symbols,
            alphabets_max_symbols=self.alphabets_max_symbols,
            alphabets_step=self.alphabets_step,
            dilations_min_dilation=self.dilations_min_dilation,
            dilations_max_dilation=self.dilations_max_dilation,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
            n_jobs=self.n_jobs,
            n_jobs_numba=self.n_jobs_numba,
            transformer_weights=self.transformer_weights,
            pipeline_objects=pipeline_objects,
            complexity=self.complexity,
        )
        self.pipe_.fit(X)
        return self

    def _transform(self, X, y=None):
        return self.pipe_.transform(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return the "default" set.

        Returns
        -------
        dict or list of dict
            Parameters to create testing instances of the class.
            Each dict contains parameters to construct an "interesting" test instance,
            i.e., `MyClass(**params)` or `MyClass(**params[i])` creates a valid test
            instance. `create_test_instance` uses the first (or only) dictionary in
            `params`.
        """
        params = [{"densify": False}, {"densify": True}]
        return params


def _build_pipeline_auto(
    time_series_min_length: int,
    time_series_max_length: int,
    window_size_min_window_size=4,
    window_size_max_window_size=None,
    word_lengths_n_word_lengths=4,
    alphabets_min_symbols=3,
    alphabets_max_symbols=4,
    alphabets_step=1,
    dilations_min_dilation=1,
    dilations_max_dilation=None,
    min_window_to_signal_std_ratio: float = 0.0,
    n_jobs=1,
    n_jobs_numba=1,
    transformer_weights=None,
    pipeline_objects: Optional[Sequence[tuple]] = None,
    complexity: Literal["quadratic", "linear"] = "quadratic",
):
    configs = _heuristic_function_sax(
        time_series_min_length=time_series_min_length,
        time_series_max_length=time_series_max_length,
        window_size_min_window_size=window_size_min_window_size,
        window_size_max_window_size=window_size_max_window_size,
        word_lengths_n_word_lengths=word_lengths_n_word_lengths,
        alphabets_min_symbols=alphabets_min_symbols,
        alphabets_max_symbols=alphabets_max_symbols,
        alphabets_step=alphabets_step,
        dilations_min_dilation=dilations_min_dilation,
        dilations_max_dilation=dilations_max_dilation,
        complexity=complexity,
    )

    return (
        _build_pipeline(
            configs=configs,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            n_jobs=n_jobs,
            n_jobs_numba=n_jobs_numba,
            transformer_weights=transformer_weights,
            pipeline_objects=pipeline_objects,
        ),
        configs,
    )


def _build_pipeline(
    configs,
    min_window_to_signal_std_ratio: float = 0.0,
    n_jobs_numba=1,
    n_jobs=1,
    transformer_weights=None,
    pipeline_objects: Optional[Sequence[tuple]] = None,
):
    transformers = list()
    if pipeline_objects is None:
        pipeline_objects = list()
    for config in configs:
        # alphabet_size, window_size, word_length, dilation, stride
        borf = IndividualBORF(
            **config,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            n_jobs=n_jobs_numba,
        )
        transformer = make_pipeline_sklearn(
            borf, *[obj(**kwargs) for obj, kwargs in pipeline_objects]
        )
        transformers.append(transformer)
    union = FeatureUnion(
        transformer_list=[(str(i), transformers[i]) for i in range(len(transformers))],
        n_jobs=n_jobs,
        transformer_weights=transformer_weights,
    )
    return union


class IndividualBORF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size=4,
        dilation=1,
        alphabet_size: int = 3,
        word_length: int = 2,
        stride: int = 1,
        min_window_to_signal_std_ratio: float = 0.0,
        n_jobs: int = 1,
        prefix="",
    ):
        self.window_size = window_size
        self.dilation = dilation
        self.word_length = word_length
        self.stride = stride
        self.alphabet_size = alphabet_size
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.prefix = prefix
        self.n_jobs = n_jobs
        self.n_words = _convert_to_base_10(
            _array_to_int(np.full(self.word_length, self.alphabet_size - 1)) + 1,
            base=self.alphabet_size,
        )
        _set_n_jobs_numba(n_jobs=self.n_jobs)

        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        import sparse  # FIXME: can we move this outside for better performance?

        shape_ = (len(X), len(X[0]), self.n_words)
        out = _transform_sax_patterns(
            panel=X,
            window_size=self.window_size,
            dilation=self.dilation,
            alphabet_size=self.alphabet_size,
            word_length=self.word_length,
            stride=self.stride,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
        )
        # ts_idx, signal_idx, words, count
        return sparse.COO(coords=out[:, :3].T, data=out[:, -1].T, shape=shape_)


class _ZeroColumnsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, axis=0):
        self.axis = axis

        self.n_original_columns_ = None
        self.columns_to_keep_ = None

        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        self.n_original_columns_ = X.shape[1]
        self.columns_to_keep_ = np.argwhere(X.any(axis=self.axis)).ravel()
        return self

    def transform(self, X):
        return X[..., self.columns_to_keep_]


class _ReshapeTo2D(BaseEstimator, TransformerMixin):
    def __init__(self, keep_unraveled_index=False):
        self.keep_unraveled_index = keep_unraveled_index
        # shape: (n_flattened_features, 2) -> flattened index -> (dimension, word)
        self.unraveled_index_ = None
        self.original_shape_ = None

        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        self.original_shape_ = X.shape
        if self.keep_unraveled_index:
            self.unraveled_index_ = np.hstack(
                [np.unravel_index(np.arange(np.prod(X.shape[1:])), X.shape[1:])]
            ).T
        return self

    def transform(self, X):
        return X.reshape((X.shape[0], -1))


class _ToScipySparse(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_scipy_sparse()


class _ToDense(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


@nb.njit(fastmath=True, cache=True)
def _convert_to_base_10(number, base):
    result = 0
    multiplier = 1

    while number > 0:
        digit = number % 10
        result += digit * multiplier
        multiplier *= base
        number //= 10

    return result


@nb.njit(cache=True)
def _array_to_int(arr):
    result = 0
    for i in range(len(arr)):
        result = result * 10 + arr[i]
    return result


def _set_n_jobs_numba(n_jobs):
    if n_jobs == -1:
        nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
    else:
        nb.set_num_threads(n_jobs)


@nb.njit(parallel=True, nogil=True, cache=True)
def _transform_sax_patterns(
    panel,
    window_size,
    word_length,
    alphabet_size,
    stride,
    dilation,
    min_window_to_signal_std_ratio=0.0,
):
    bins = _get_norm_bins(alphabet_size=alphabet_size)
    n_signals = len(panel[0])
    n_ts = len(panel)
    iterations = n_ts * n_signals
    counts = np.zeros(iterations + 1, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = _ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx])
        signal = signal[~np.isnan(signal)]
        if not _are_window_size_and_dilation_compatible_with_signal_length(
            window_size, dilation, signal.size
        ):
            continue
        counts[i + 1] = len(
            _new_transform_single_conf(
                a=signal,
                ts_idx=ts_idx,
                signal_idx=signal_idx,
                window_size=window_size,
                word_length=word_length,
                alphabet_size=alphabet_size,
                bins=bins,
                dilation=dilation,
                stride=stride,
                min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            )
        )
    cum_counts = np.cumsum(counts)
    n_rows = np.sum(counts)
    shape = (n_rows, 4)
    out = np.empty(shape, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = _ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx])
        signal = signal[~np.isnan(signal)]
        if not _are_window_size_and_dilation_compatible_with_signal_length(
            window_size, dilation, signal.size
        ):
            continue
        out_ = _new_transform_single_conf(
            a=signal,
            ts_idx=ts_idx,
            signal_idx=signal_idx,
            window_size=window_size,
            word_length=word_length,
            alphabet_size=alphabet_size,
            bins=bins,
            dilation=dilation,
            stride=stride,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        )
        out[cum_counts[i] : cum_counts[i + 1], :] = out_
    return out


@nb.njit(cache=True)
def _new_transform_single_conf(
    a,
    ts_idx,
    signal_idx,
    window_size,
    word_length,
    alphabet_size,
    bins,
    dilation,
    stride=1,
    min_window_to_signal_std_ratio=0.0,
):
    words, counts = _new_transform_single(
        a=a,
        window_size=window_size,
        word_length=word_length,
        alphabet_size=alphabet_size,
        bins=bins,
        dilation=dilation,
        stride=stride,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
    )
    ts_idxs = np.full(len(words), ts_idx)
    signal_idxs = np.full(len(words), signal_idx)
    return np.column_stack((ts_idxs, signal_idxs, words, counts))


@nb.njit(cache=True)
def _new_transform_single(
    a,
    window_size,
    word_length,
    alphabet_size,
    bins,
    dilation,
    stride=1,
    min_window_to_signal_std_ratio=0.0,
):
    sax_words = _sax(
        a=a,
        window_size=window_size,
        word_length=word_length,
        bins=bins,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        dilation=dilation,
        stride=stride,
    )
    sax_words = _sax_words_to_int(sax_words, alphabet_size)
    return _unique(sax_words)


@nb.njit(cache=True)
def _ndindex_2d_array(idx, dim2_shape):
    row_idx = idx // dim2_shape
    col_idx = idx % dim2_shape
    return row_idx, col_idx


@nb.njit(cache=True)
def _get_norm_bins(alphabet_size: int, mu=0, std=1):
    bins = []
    for i in np.linspace(0, 1, alphabet_size + 1)[1:-1]:
        bins.append(_ppf(i, mu, std))
    return np.array(bins)


@nb.njit(fastmath=True, cache=True)
def _erfinv(x: float) -> float:
    w = -np.log((1 - x) * (1 + x))
    if w < 5:
        w = w - 2.5
        p = 2.81022636e-08
        p = 3.43273939e-07 + p * w
        p = -3.5233877e-06 + p * w
        p = -4.39150654e-06 + p * w
        p = 0.00021858087 + p * w
        p = -0.00125372503 + p * w
        p = -0.00417768164 + p * w
        p = 0.246640727 + p * w
        p = 1.50140941 + p * w
    else:
        w = np.sqrt(w) - 3
        p = -0.000200214257
        p = 0.000100950558 + p * w
        p = 0.00134934322 + p * w
        p = -0.00367342844 + p * w
        p = 0.00573950773 + p * w
        p = -0.0076224613 + p * w
        p = 0.00943887047 + p * w
        p = 1.00167406 + p * w
        p = 2.83297682 + p * w
    return p * x


@nb.njit(cache=True)
def _ppf(x, mu=0, std=1):
    return mu + np.sqrt(2) * _erfinv(2 * x - 1) * std


@nb.njit(fastmath=True, cache=True)
def _are_window_size_and_dilation_compatible_with_signal_length(
    window_size, dilation, signal_length
):
    if window_size + (window_size - 1) * (dilation - 1) <= signal_length:
        return True
    else:
        return False


@nb.njit(cache=True)
def _sax(
    a,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    n_windows = _get_n_windows(
        sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride
    )
    n_windows_moving = _get_n_windows(
        sequence_size=a.size, window_size=window_size, dilation=dilation
    )
    global_std = np.std(a)
    if global_std == 0:
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    seg_size = window_size // word_length
    n_windows = _get_n_windows(
        sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride
    )
    n_segments = _get_n_windows(
        sequence_size=a.size, window_size=seg_size, dilation=dilation
    )
    segment_means = np.full(n_segments, np.nan)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    for d in range(dilation):
        window_means[d::dilation] = _move_mean(a[d::dilation], window_size)[
            window_size - 1 :
        ]
        window_stds[d::dilation] = _move_std(a[d::dilation], window_size)[
            window_size - 1 :
        ]
        segment_means[d::dilation] = _move_mean(a[d::dilation], seg_size)[
            seg_size - 1 :
        ]
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = _zscore_threshold(
                a=segment_means[(i * stride) + (j * seg_size * dilation)],
                mu=window_means[i * stride],
                sigma=window_stds[i * stride],
                sigma_global=global_std,
                sigma_threshold=min_window_to_signal_std_ratio,
            )
    return np.digitize(out, bins).astype(np.uint8)


@nb.njit(fastmath=True, cache=True)
def _get_n_windows(sequence_size, window_size, dilation=1, stride=1, padding=0):
    return 1 + math.floor(
        (sequence_size + 2 * padding - window_size - (dilation - 1) * (window_size - 1))
        / stride
    )


@nb.njit(fastmath=True, cache=True)
def _zscore_threshold(
    a: float, mu: float, sigma: float, sigma_global: float, sigma_threshold: float
) -> float:
    if sigma_global == 0:
        return 0
    if sigma / sigma_global < sigma_threshold:
        return 0
    return _zscore(a=a, mu=mu, sigma=sigma)


@nb.njit(fastmath=True, cache=True)
def _zscore(a: float, mu: float, sigma: float) -> float:
    if sigma == 0:
        return 0
    return (a - mu) / sigma


@nb.njit(fastmath=True, cache=True)
def _move_mean(a, window_width):
    out = np.empty_like(a)
    asum = 0.0
    count = 0

    # Calculating the initial moving window sum
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count

    # Moving window
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / window_width
    return out


@nb.njit(fastmath=True, cache=True)
def _move_std(a, window_width, ddof=0):
    out = np.empty_like(a, dtype=np.float64)

    # Initial mean and variance calculation
    mean = 0.0
    M2 = 0.0
    for i in range(window_width):
        delta = a[i] - mean
        mean += delta / (i + 1)
        delta2 = a[i] - mean
        M2 += delta * delta2

    # Adjusting for degrees of freedom
    if window_width - ddof > 0:
        variance = M2 / (window_width - ddof)
    else:
        variance = 0  # Avoid division by zero

    out[window_width - 1] = np.sqrt(max(variance, 0))

    # Moving window
    for i in range(window_width, len(a)):
        x0 = a[i - window_width]
        xn = a[i]

        new_avg = mean + (xn - x0) / window_width

        # Update the variance using the new degrees of freedom
        if window_width - ddof > 0:
            new_var = variance + (xn - new_avg + x0 - mean) * (xn - x0) / (
                window_width - ddof
            )
        else:
            new_var = 0  # Avoid division by zero

        out[i] = np.sqrt(max(new_var, 0))  # TODO: investigate negative variance

        mean = new_avg
        variance = new_var

    out[: window_width - 1] = np.nan

    return out


@nb.njit(fastmath=True, cache=True)
def _sax_words_to_int(arrays, base):
    out = np.empty(arrays.shape[0], dtype=np.int64)
    for i in range(arrays.shape[0]):
        out[i] = _array_to_int_new_base(arrays[i], base)
    return out


@nb.njit(fastmath=True, cache=True)
def _array_to_int_new_base(array, base):
    word_length = array.shape[0]
    result = 0
    for i in range(0, word_length, 1):
        result += array[i] * base ** (word_length - i - 1)
    return result


@nb.njit(cache=True)
def _length(a):
    a = int(np.ceil(np.log2(a)))
    a = 2 << a
    return a


@nb.njit(cache=True)
def _hash_function(v):
    byte_mask = np.uint64(255)
    bs = np.uint64(v)
    x1 = (bs) & byte_mask
    x2 = (bs >> np.uint64(8)) & byte_mask
    x3 = (bs >> np.uint64(16)) & byte_mask
    x4 = (bs >> np.uint64(24)) & byte_mask
    x5 = (bs >> np.uint64(32)) & byte_mask
    x6 = (bs >> np.uint64(40)) & byte_mask
    x7 = (bs >> np.uint64(48)) & byte_mask
    x8 = (bs >> np.uint64(56)) & byte_mask

    FNV_primer = np.uint64(1099511628211)
    FNV_bias = np.uint64(14695981039346656037)
    h = FNV_bias
    h = h * FNV_primer
    h = h ^ x1
    h = h * FNV_primer
    h = h ^ x2
    h = h * FNV_primer
    h = h ^ x3
    h = h * FNV_primer
    h = h ^ x4
    h = h * FNV_primer
    h = h ^ x5
    h = h * FNV_primer
    h = h ^ x6
    h = h * FNV_primer
    h = h ^ x7
    h = h * FNV_primer
    h = h ^ x8
    return h


@nb.njit(cache=True)
def _make_hash_table(ar):
    a = _length(len(ar))
    mask = np.uint64(a - 1)

    uniques = np.empty(a, dtype=ar.dtype)
    uniques_cnt = np.zeros(a, dtype=np.int_)
    return uniques, uniques_cnt, a, mask


@nb.njit(cache=True)
def _set_item(uniques, uniques_cnt, mask, h, v, total, miss_hits, weight):
    index = h & mask
    while True:
        if uniques_cnt[index] == 0:
            # insert new
            uniques_cnt[index] += weight
            uniques[index] = v
            total += 1
            break
        elif uniques[index] == v:
            uniques_cnt[index] += weight
            break
        else:
            miss_hits += 1
            index += np.uint64(1)
            index = index & mask
    return total, miss_hits


@nb.njit(cache=True)
def _concrete(ar, uniques, uniques_cnt, a, total):
    # flush the results in a concrete array
    uniques_ = np.empty(total, dtype=ar.dtype)
    uniques_cnt_ = np.empty(total, dtype=np.int_)
    t = 0
    for i in range(a):
        if uniques_cnt[i] > 0:
            uniques_[t] = uniques[i]
            uniques_cnt_[t] = uniques_cnt[i]
            t += 1
    return uniques_, uniques_cnt_


@nb.njit(cache=True)
def _unique(ar):
    # https://github.com/lhprojects/blog/blob/master/_posts/JupyterNotebooks/HashUnique.ipynb
    uniques, uniques_cnt, l, mask = _make_hash_table(ar)
    total = 0
    miss_hits = 0
    for v in ar:
        h = _hash_function(v)
        total, miss_hits = _set_item(
            uniques, uniques_cnt, mask, h, v, total, miss_hits, 1
        )
    uniques_, uniques_cnt_ = _concrete(ar, uniques, uniques_cnt, l, total)
    return uniques_, uniques_cnt_


def _get_borf_params(
    time_series_min_length,
    time_series_max_length,
    window_size_min_window_size=4,
    window_size_max_window_size=None,
    window_size_power=2,
    word_lengths_n_word_lengths=4,
    strides_n_strides=1,
    alphabets_mean_min_symbols=2,
    alphabets_mean_max_symbols=3,
    alphabets_mean_step=1,
    alphabets_slope_min_symbols=3,
    alphabets_slope_max_symbols=4,
    alphabets_slope_step=1,
    dilations_min_dilation=1,
    dilations_max_dilation=None,
):
    params = {}
    params["window_sizes"] = _get_window_sizes(
        m_min=time_series_min_length,
        m_max=time_series_max_length,
        min_window_size=window_size_min_window_size,
        max_window_size=window_size_max_window_size,
        power=window_size_power,
    ).tolist()

    params["word_lengths"] = _get_word_lengths(
        n_word_lengths=word_lengths_n_word_lengths,
        start=0,
    ).tolist()

    params["dilations"] = _get_dilations(
        max_length=time_series_max_length,
        min_dilation=dilations_min_dilation,
        max_dilation=dilations_max_dilation,
    ).tolist()
    params["strides"] = _get_strides(n_strides=strides_n_strides).tolist()
    params["alphabet_sizes_slope"] = _get_alphabet_sizes(
        min_symbols=alphabets_slope_min_symbols,
        max_symbols=alphabets_slope_max_symbols,
        step=alphabets_slope_step,
    ).tolist()
    params["alphabet_sizes_mean"] = _get_alphabet_sizes(
        min_symbols=alphabets_mean_min_symbols,
        max_symbols=alphabets_mean_max_symbols,
        step=alphabets_mean_step,
    ).tolist()
    return params


@nb.njit(cache=True)
def _is_empty(a) -> bool:
    return a.size == 0


def _get_window_sizes(m_min, m_max, min_window_size=4, max_window_size=None, power=2):
    if max_window_size is None:
        max_window_size = m_max
    m = 2
    windows = list()
    windows_min = list()
    while m <= max_window_size:
        if m < min_window_size:
            windows_min.append(m)
        else:
            windows.append(m)
        m = int(m * power)
    windows = np.array(windows)
    windows_min = np.array(windows_min[1:])
    if not _is_empty(windows_min):
        if m_min <= windows_min.max() * power:
            windows = np.concatenate([windows_min, windows])
    return windows.astype(int)


def _get_word_lengths(n_word_lengths=4, start=0):
    return np.array([2**i for i in range(start, n_word_lengths + start)])


def _get_dilations(max_length, min_dilation=1, max_dilation=None):
    dilations = list()
    max_length_log2 = np.log2(max_length)
    if max_dilation is None:
        max_dilation = max_length_log2
    start = min_dilation
    while start <= max_dilation:
        dilations.append(start)
        start *= 2
    return np.array(dilations)


def _get_strides(n_strides=1):
    return np.arange(1, n_strides + 1)


def _get_alphabet_sizes(min_symbols=3, max_symbols=4, step=1):
    return np.arange(min_symbols, max_symbols, step)


def _generate_sax_parameters_configurations(
    window_sizes,
    strides,
    dilations,
    word_lengths,
    alphabet_sizes_mean,
    alphabet_sizes_slope,
):
    parameters = list(
        itertools.product(
            *[
                window_sizes,
                strides,
                dilations,
                word_lengths,
                alphabet_sizes_mean,
                alphabet_sizes_slope,
            ]
        )
    )
    cleaned_parameters = list()
    for parameter in parameters:
        (
            window_size,
            stride,
            dilation,
            word_length,
            alphabet_mean,
            alphabet_slope,
        ) = _extract_parameters_from_args(parameter)
        if word_length > window_size:  # word_length cannot be greater than window_size
            continue
        if (
            alphabet_slope <= 1 and word_length == 1
        ):  # if alphabet_slope <= 1, word_length=1 is useless
            continue
        cleaned_parameters.append(
            dict(
                window_size=window_size,
                stride=stride,
                dilation=dilation,
                word_length=word_length,
                alphabet_size_mean=alphabet_mean,
                alphabet_size_slope=alphabet_slope,
            )
        )
    return cleaned_parameters


@nb.njit(cache=True)
def _is_valid_windowing(sequence_size: int, window_size: int, dilation: int) -> bool:
    if (
        sequence_size < window_size * dilation
    ):  # if window_size * dilation exceeds the length of the sequence
        return False
    else:
        return True


def _clean_sax_parameters_configurations(parameters, max_length):
    cleaned_parameters = list()
    for parameter in parameters:
        window_size = parameter["window_size"]
        dilation = parameter["dilation"]
        word_length = parameter["word_length"]
        alphabet_mean = parameter["alphabet_size_mean"]
        alphabet_slope = parameter["alphabet_size_slope"]
        stride = parameter["stride"]
        if not _is_valid_windowing(
            window_size=window_size, sequence_size=max_length, dilation=dilation
        ):
            continue
        cleaned_parameters.append(
            dict(
                window_size=window_size,
                stride=stride,
                dilation=dilation,
                word_length=word_length,
                alphabet_size_mean=alphabet_mean,
                alphabet_size_slope=alphabet_slope,
            )
        )
    return cleaned_parameters


def _sax_parameters_configurations_linear_strides(parameters):
    new_parameters = list()
    for parameter in parameters:
        window_size = parameter["window_size"]
        dilation = parameter["dilation"]
        word_length = parameter["word_length"]
        alphabet_mean = parameter["alphabet_size_mean"]
        alphabet_slope = parameter["alphabet_size_slope"]
        parameter = dict(
            window_size=window_size,
            stride=word_length,
            dilation=dilation,
            word_length=word_length,
            alphabet_size_mean=alphabet_mean,
            alphabet_size_slope=alphabet_slope,
        )
        new_parameters.append(parameter)
    return new_parameters


def _extract_parameters_from_args(parameter):
    window_size = parameter[0]
    stride = parameter[1]
    dilation = parameter[2]
    word_length = parameter[3]
    alphabet_mean = parameter[4]
    alphabet_slope = parameter[5]
    return window_size, stride, dilation, word_length, alphabet_mean, alphabet_slope


def _heuristic_function_1dsax(
    time_series_min_length,
    time_series_max_length,
    window_size_min_window_size=4,
    window_size_max_window_size=None,
    window_size_power=2,
    word_lengths_n_word_lengths=3,
    strides_n_strides=1,
    alphabets_mean_min_symbols=2,
    alphabets_mean_max_symbols=3,
    alphabets_mean_step=1,
    alphabets_slope_min_symbols=2,
    alphabets_slope_max_symbols=3,
    alphabets_slope_step=1,
    dilations_min_dilation=1,
    dilations_max_dilation=None,
    complexity: Literal["quadratic", "linear"] = "quadratic",
):
    params = _get_borf_params(
        time_series_min_length=time_series_min_length,
        time_series_max_length=time_series_max_length,
        window_size_min_window_size=window_size_min_window_size,
        window_size_max_window_size=window_size_max_window_size,
        window_size_power=window_size_power,
        word_lengths_n_word_lengths=word_lengths_n_word_lengths,
        strides_n_strides=strides_n_strides,
        alphabets_slope_min_symbols=alphabets_slope_min_symbols,
        alphabets_slope_max_symbols=alphabets_slope_max_symbols,
        alphabets_slope_step=alphabets_slope_step,
        alphabets_mean_min_symbols=alphabets_mean_min_symbols,
        alphabets_mean_max_symbols=alphabets_mean_max_symbols,
        alphabets_mean_step=alphabets_mean_step,
        dilations_min_dilation=dilations_min_dilation,
        dilations_max_dilation=dilations_max_dilation,
    )

    params_list = _generate_sax_parameters_configurations(
        window_sizes=params["window_sizes"],
        strides=params["strides"],
        dilations=params["dilations"],
        word_lengths=params["word_lengths"],
        alphabet_sizes_mean=params["alphabet_sizes_mean"],
        alphabet_sizes_slope=params["alphabet_sizes_slope"],
    )

    cleaned_params_list = _clean_sax_parameters_configurations(
        parameters=params_list, max_length=time_series_max_length
    )

    if complexity == "linear":
        cleaned_params_list = _sax_parameters_configurations_linear_strides(
            parameters=cleaned_params_list
        )

    return cleaned_params_list


def _heuristic_function_sax(
    time_series_min_length,
    time_series_max_length,
    window_size_min_window_size=4,
    window_size_max_window_size=None,
    window_size_power=2,
    word_lengths_n_word_lengths=4,
    strides_n_strides=1,
    alphabets_min_symbols=2,
    alphabets_max_symbols=3,
    alphabets_step=1,
    dilations_min_dilation=1,
    dilations_max_dilation=None,
    complexity: Literal["quadratic", "linear"] = "quadratic",
):
    configs = _heuristic_function_1dsax(
        time_series_min_length=time_series_min_length,
        time_series_max_length=time_series_max_length,
        window_size_min_window_size=window_size_min_window_size,
        window_size_max_window_size=window_size_max_window_size,
        window_size_power=window_size_power,
        word_lengths_n_word_lengths=word_lengths_n_word_lengths,
        strides_n_strides=strides_n_strides,
        alphabets_slope_min_symbols=0,
        alphabets_slope_max_symbols=1,
        alphabets_slope_step=1,
        alphabets_mean_min_symbols=alphabets_min_symbols,
        alphabets_mean_max_symbols=alphabets_max_symbols,
        alphabets_mean_step=alphabets_step,
        dilations_min_dilation=dilations_min_dilation,
        dilations_max_dilation=dilations_max_dilation,
        complexity=complexity,
    )
    for config in configs:
        config.pop("alphabet_size_slope")
        config["alphabet_size"] = config.pop("alphabet_size_mean")
    return configs
