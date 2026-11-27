"""Catch22 features.

A transformer for the Catch22 features.
"""

__maintainer__ = []
__all__ = ["Catch22"]

import math
import warnings
from weakref import WeakSet

import numpy as np
from joblib import Parallel, delayed
from numba import njit

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.numba.general import (
    _pow2_scale,
    z_normalise_series_2d,
    z_normalise_series_3d,
)
from aeon.utils.numba.stats import mean, numba_max, numba_min
from aeon.utils.validation import check_n_jobs

# numba's fastmath without its no-NaN and no-inf assumptions: features return NaN
# when they cannot be estimated, and some kernels use inf as a sentinel. Values
# can differ from catch22's C in the last bits, so a feature decided by an exact
# tie (e.g. on a series with repeated windows) can resolve differently.
_FASTMATH = {"reassoc", "contract", "arcp", "nsz", "afn"}

# Feature names as in catch22 v1.0. The order is aeon's, not catch22's, so that
# selecting features by index is unchanged.
feature_names = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_FirstCrossing_acf_1e_pointOfCrossing",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_5bin_ami2",
    "CO_trev_1_num",
    "MD_pNN_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_quantile_3_ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_taurat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_fft_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_fft_wq_50",
    "FC_LocalSimple_mean3_stde",
]

feature_names_short = [
    "mode_5",
    "mode_10",
    "acf_timescale",
    "acf_first_min",
    "ami2",
    "trev",
    "high_fluctuation",
    "stretch_high",
    "transition_variance",
    "periodicity",
    "embedding_dist",
    "ami_timescale",
    "whiten_timescale",
    "outlier_timing_pos",
    "outlier_timing_neg",
    "low_freq_power",
    "stretch_decreasing",
    "entropy_pairs",
    "rs_range",
    "dfa",
    "median_freq",
    "forecast_error",
]

# TODO remove the catch22 v0.x names in v1.8.0
legacy_feature_names = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]
_legacy_short_names = {"transition_matrix": 8, "centroid_freq": 20}
_WARNED_LEGACY_NAMES = WeakSet()

# catch24's two extra features, by aeon's and catch22's names
_catch24_names = {
    "Mean": 22,
    "StandardDeviation": 23,
    "DN_mean": 22,
    "DN_Spread_std": 23,
    "mean": 22,
    "std": 23,
}

# no precomputed transition_variance filters (see _transition_filters)
_NO_FILTERS = np.empty(0, dtype=np.float64)
_NO_FILTER_START = np.empty(0, dtype=np.int64)


def _mask_value(feature, n):
    """Return catch22's value for a feature that cannot be estimated.

    A constant series is treated as the limit of an infinitely slow, perfectly
    predictable series, and a failure as the absence of the structure the feature
    looks for, which places it at one end of the feature's range.
    """
    if feature == 2:
        # the autocorrelation never decays
        return n - 1.0
    if feature == 3:
        # no minimum of the autocorrelation
        return float(n)
    if feature == 8:
        # perfectly predictable transitions between three equiprobable symbols
        return 1.0 / 9.0
    if feature == 10:
        # distances maximally far from exponential (all distances equal)
        range_term = 1.0 / (math.sqrt(2.0) * 0.001)
        mean_term = (
            (n - 2) / (math.sqrt(2.0) * math.sqrt((n - 1.0) / n)) if n > 2 else 0.0
        )
        return range_term + mean_term
    if feature == 11:
        # no minimum of the automutual information within the lags analysed
        return float(min(40, (n + 1) // 2))
    if feature in (15, 18, 19):
        return 1.0
    return 0.0


class Catch22(BaseCollectionTransformer):
    """Canonical Time-series Characteristics (Catch22).

    Overview: Input n series with d channels of length m.
    Transforms series into the 22 Catch22 [1]_ features extracted from the hctsa [2]_
    toolbox. The features follow catch22 v1.2.0, in which each feature reproduces its
    hctsa v3.0.0 definition.

    Parameters
    ----------
    features : int/str or List of int/str, default="all"
        The Catch22 features to extract by feature index, feature name as a str or as a
        list of names or indices for multiple features. If "all", all features are
        extracted.
        Valid features are as follows:
            ["DN_HistogramMode_5", "DN_HistogramMode_10",
            "CO_FirstCrossing_acf_1e_pointOfCrossing", "CO_FirstMin_ac",
            "CO_HistogramAMI_even_5bin_ami2", "CO_trev_1_num", "MD_pNN_pnn40",
            "SB_BinaryStats_mean_longstretch1",
            "SB_TransitionMatrix_quantile_3_ac_sumdiagcov",
            "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi", "FC_LocalSimple_mean1_taurat",
            "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd",
            "SP_Summaries_fft_area_5_1", "SB_BinaryStats_diff_longstretch0",
            "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_fft_wq_50",
            "FC_LocalSimple_mean3_stde"]
        Shortened:
            ["mode_5", "mode_10", "acf_timescale", "acf_first_min",
            "ami2", "trev", "high_fluctuation",
            "stretch_high", "transition_variance",
            "periodicity", "embedding_dist",
            "ami_timescale", "whiten_timescale",
            "outlier_timing_pos", "outlier_timing_neg",
            "low_freq_power", "stretch_decreasing",
            "entropy_pairs", "rs_range",
            "dfa",
            "median_freq", "forecast_error"]

        The feature names from catch22 v0.x (e.g. "CO_f1ecac" or "centroid_freq")
        are deprecated and will be removed in v1.8.0. They select the same position,
        which now holds the catch22 v1.0 feature.
    catch24 : bool, default=False
        Extract the mean and standard deviation as well as the 22 Catch22 features if
        true. If a List of specific features to extract is provided, "Mean" and/or
        "StandardDeviation" (or catch22's "mean" and "std") must be added to the List
        to extract these features.
    outlier_norm : bool, default=True
        If True, the two outlier Catch22 features are computed on each time series
        z-normalised as for ``normalise``, as catch22 computes all features on
        z-normalised series. Without it, the outlier features of a series with
        values far from zero are NaN (masked if ``mask_nans`` is True). Has no
        effect if ``normalise`` is True.
    normalise : bool, default=False
        If True, each time series is z-normalised before the 22 Catch22 features are
        computed, as catch22 recommends. As elsewhere in aeon, a series is divided by
        its population standard deviation, or only centred if that is at most 1e-8.
        catch22 itself divides by the sample standard deviation, so the features
        that depend on scale differ slightly from catch22's on short series. If
        False, the features are computed on the series as given, and eight of them
        then depend on its offset or scale: "DN_HistogramMode_5",
        "DN_HistogramMode_10", "CO_trev_1_num", "MD_pNN_pnn40",
        "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
        "SP_Summaries_fft_area_5_1" and "FC_LocalSimple_mean3_stde". With catch24,
        the mean and standard deviation are always those of the series as given.
    mask_nans : bool, default=True
        Replace a feature that cannot be estimated (e.g. for a constant series, or
        "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1" on a series shorter than 79
        values) with a fixed value at the appropriate end of its range, as catch22
        does. If False, such features are NaN.
    replace_nans : bool, default=False
        Replace NaN or inf values from the Catch22 transform with 0.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `transform`. Requires multiple input
        cases. ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    See Also
    --------
    Catch22Classifier

    Notes
    -----
    Original Catch22 package implementations:
    https://github.com/DynamicsAndNeuralSystems/Catch22

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/transformers/Catch22.java

    References
    ----------
    .. [1] Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., &
    Jones, N. S. (2019). catch22: Canonical time-series characteristics. Data Mining
    and Knowledge Discovery, 33(6), 1821-1852.
    .. [2] Fulcher, B. D., Little, M. A., & Jones, N. S. (2013). Highly comparative
    time-series analysis: the empirical structure of time series and their methods.
    Journal of the Royal Society Interface, 10(83), 20130048.

    Examples
    --------
    >>> from aeon.transformations.collection.feature_based import Catch22
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X = make_example_3d_numpy(n_cases=4, n_channels=1, n_timepoints=10,
    ...                           random_state=0, return_y=False)
    >>> tnf = Catch22(replace_nans=True)
    >>> tnf.fit(X)
    Catch22(...)
    >>> print(tnf.transform(X)[0])
    [1.15639493 1.31700564 0.56622771 2.         0.67021592 0.23385358
     1.         2.         0.00823045 0.         0.41695174 3.
     0.91537551 0.7        0.2        0.03988775 3.         2.20985854
     1.         1.         1.96349541 0.551667  ]
    """

    _tags = {
        "output_data_type": "Tabular",
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:unequal_length": True,
        "capability:multivariate": True,
        "capability:multithreading": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        features="all",
        catch24=False,
        outlier_norm=True,
        normalise=False,
        mask_nans=True,
        replace_nans=False,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.normalise = normalise
        self.mask_nans = mask_nans
        self.replace_nans = replace_nans
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

        # TODO remove with the catch22 v0.x names in v1.8.0
        # warn once per instance: reset() calls __init__ again on every fit
        legacy = _legacy_names_in(features)
        if legacy and self not in _WARNED_LEGACY_NAMES:
            _WARNED_LEGACY_NAMES.add(self)
            renamed = [feature_names[_feature_index(f, True)] for f in legacy]
            warnings.warn(
                f"The catch22 v0.x feature names {legacy} are deprecated and will be "
                f"removed in v1.8.0. They now select the catch22 v1.0 features "
                f"{renamed}; use those names instead.",
                FutureWarning,
                stacklevel=2,
            )

    def _transform(self, X, y=None):
        """Transform X into the catch22 features.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_cases], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i

        Returns
        -------
        Xt : array-like, shape = [n_cases, num_features*n_channels]
            The catch22 features for each channel.
        """
        n_cases = len(X)

        f_idx = _verify_features(self.features, self.catch24)
        n_jobs = check_n_jobs(self.n_jobs)

        # catch24's mean and standard deviation are those of the series as given
        X_given = X
        if self.normalise:
            X = _normalise(X)
        # the series the two outlier features are computed on
        X_outlier = X
        if self.outlier_norm and not self.normalise and (13 in f_idx or 14 in f_idx):
            X_outlier = _normalise(X)

        features = [
            Catch22._DN_HistogramMode_5,
            Catch22._DN_HistogramMode_10,
            Catch22._CO_FirstCrossing_acf_1e_pointOfCrossing,
            Catch22._CO_FirstMin_ac,
            Catch22._CO_HistogramAMI_even_5bin_ami2,
            Catch22._CO_trev_1_num,
            Catch22._MD_pNN_pnn40,
            Catch22._SB_BinaryStats_mean_longstretch1,
            Catch22._SB_TransitionMatrix_quantile_3_ac_sumdiagcov,
            Catch22._PD_PeriodicityWang_th0_01,
            Catch22._CO_Embed2_Dist_tau_d_expfit_meandiff,
            Catch22._IN_AutoMutualInfoStats_40_gaussian_fmmi,
            Catch22._FC_LocalSimple_mean1_taurat,
            Catch22._DN_OutlierInclude_p_001_mdrmd,
            Catch22._DN_OutlierInclude_n_001_mdrmd,
            Catch22._SP_Summaries_fft_area_5_1,
            Catch22._SB_BinaryStats_diff_longstretch0,
            Catch22._SB_MotifThree_quantile_hh,
            Catch22._SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
            Catch22._SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
            Catch22._SP_Summaries_fft_wq_50,
            Catch22._FC_LocalSimple_mean3_stde,
        ]

        # The two power-spectrum features (indices 15 and 20) each need
        # np.fft.fft of the mean-centred series. np.fft.fft has a high fixed
        # per-call cost on short interval series, so compute it once for the
        # whole batch (all cases share a length) instead of once per case. The
        # result matches the per-case FFT to within rounding (the batch's means
        # can differ from the per-case mean() in the last bit).
        fft_cache = self._spectrum_fft_cache(X, f_idx, n_cases)
        # The autocorrelation features (2, 3, 8, 10, 12) all need the
        # normalised autocorrelation of the series. Compute it for the whole
        # batch with two batched np.fft calls instead of a hand-written
        # radix-2 FFT per series. NOT bit-identical: pocketfft rounds
        # differently in the last bits, so index-valued outputs can flip at
        # exact fp ties (rare).
        ac_cache = self._ac_batch_cache(X, f_idx, n_cases)
        # Twiddles for the hand-written FFT are still needed by the
        # per-series fallback (np-list input) and by feature 12's rare
        # no-crossing fallback on the differenced series.
        if ac_cache is None or 12 in f_idx:
            ac_tw, ac_nfft = self._ac_twiddle_cache(X, f_idx)
        else:
            ac_tw, ac_nfft = None, 0
        # transition_variance's low-pass filter depends only on its downsampling
        # factor, so build each filter the batch needs once. Without them (np-list
        # input) the kernel builds its filter per series.
        tv_filters, tv_start = _NO_FILTERS, _NO_FILTER_START
        if ac_cache is not None and 8 in f_idx:
            tv_filters, tv_start = _transition_filters(ac_cache, X.shape[2])

        if isinstance(X, np.ndarray) and X.ndim == 3:
            # Equal-length input: run the whole per-case dispatch loop
            # inside numba (_transform_case_numba). Each feature-kernel
            # call from Python costs several microseconds of dispatcher
            # overhead, which dominates short interval series; calling the
            # same compiled kernels from compiled code is nearly free. The
            # FFT/autocorrelation caches above are guaranteed non-None
            # whenever their features are present and unmasked, so the
            # kernel has no per-series fallbacks.
            n_channels = X.shape[1]
            tf = getattr(self, "_transform_features", None)
            if tf is not None and len(tf) == len(f_idx) * n_channels:
                keep = np.asarray(tf, dtype=np.bool_)
            else:
                keep = np.ones(len(f_idx) * n_channels, dtype=np.bool_)
            f_arr = np.asarray(f_idx, dtype=np.int64)
            # catch24's standard deviation (the sample standard deviation, as
            # catch22) stays a numpy call for exact reproducibility (numba's
            # np.std can round differently). The ascontiguousarray is
            # load-bearing: np.std along axis 2 only matches the per-series
            # np.std bit-for-bit when the last axis is contiguous, so do not
            # simplify it to np.std(X, axis=2, ddof=1).
            stds = None
            if 23 in f_idx:
                stds = np.std(np.ascontiguousarray(X_given), axis=2, ddof=1)
            means = None
            if 22 in f_idx:
                means = _row_means(
                    X_given.reshape(n_cases * n_channels, X.shape[2])
                ).reshape(n_cases, n_channels)
            # typed empty placeholders for absent caches; the matching
            # kernel branches are unreachable when a cache was not built
            no_fft = np.empty((0, 0), dtype=np.complex128)
            no_ac = np.empty((0, 0), dtype=np.float64)
            no_stat = np.empty(0, dtype=np.float64)
            tw = ac_tw if ac_tw is not None else np.empty(0, np.complex128)
            func = _transform_case_numba
            case_args = [
                (
                    X[i],
                    X_outlier[i],
                    f_arr,
                    keep,
                    no_fft if fft_cache is None else fft_cache[i],
                    no_ac if ac_cache is None else ac_cache[i],
                    tw,
                    ac_nfft,
                    no_stat if means is None else means[i],
                    no_stat if stds is None else stds[i],
                    tv_filters,
                    tv_start,
                )
                for i in range(n_cases)
            ]
        else:
            func = self._transform_case
            case_args = [
                (
                    X[i],
                    X_given[i],
                    X_outlier[i],
                    f_idx,
                    features,
                    None if fft_cache is None else fft_cache[i],
                    ac_tw,
                    ac_nfft,
                    None if ac_cache is None else ac_cache[i],
                )
                for i in range(n_cases)
            ]

        # Run cases sequentially when not parallelising: a joblib Parallel still
        # wraps every task in delayed() and copies it, which is pure overhead here
        # (this transform is called once per interval, always with n_jobs=1 inside
        # the interval forests).
        if n_jobs == 1:
            c22_list = [func(*args) for args in case_args]
        else:
            c22_list = Parallel(
                n_jobs=n_jobs, backend=self.parallel_backend, prefer="threads"
            )(delayed(func)(*args) for args in case_args)

        c22_array = np.array(c22_list)
        if self.mask_nans:
            _mask_unestimable(c22_array, X, f_idx)
        if self.replace_nans:
            c22_array = np.nan_to_num(c22_array, False, 0, 0, 0)

        return c22_array

    def _ac_twiddle_cache(self, X, f_idx):
        """Precompute the FFT twiddle table shared by the autocorrelation features.

        Features 2, 3, 8, 10 and 12 all need the autocorrelation, whose FFT twiddle
        factors depend only on the (common) series length. Build them once here and
        reuse across every case instead of rebuilding per autocorrelation call.
        Returns (None, 0) for np-list input or when no autocorrelation feature runs.
        """
        if not (isinstance(X, np.ndarray) and X.ndim == 3):
            return None, 0
        if not any(fi in (2, 3, 8, 10, 12) for fi in f_idx):
            return None, 0
        nfft = _ac_nfft(X.shape[2])
        return _ac_twiddles(nfft), nfft

    def _ac_batch_cache(self, X, f_idx, n_cases):
        """Batch the autocorrelation FFTs across all cases when they will be used.

        Features 2, 3, 8, 10 and 12 all consume the normalised autocorrelation of
        the series. The per-series path computes it with a hand-written radix-2
        numba FFT; computing the whole batch with two batched np.fft calls is far
        cheaper per series. Returns an array of shape (n_cases, n_channels,
        2 * nfft) matching _compute_autocorrelations' output per series, or None
        if no autocorrelation feature runs or the input is not an equal-length 3D
        array.

        The result is NOT bit-identical to the per-series path: pocketfft and the
        hand-written FFT differ in the last couple of bits, so downstream
        index-valued features (first zero/min crossings) can flip at exact ties.
        """
        if not (isinstance(X, np.ndarray) and X.ndim == 3):
            return None
        if not any(fi in (2, 3, 8, 10, 12) for fi in f_idx):
            return None

        n_channels = X.shape[1]

        # Respect the transform_features skip mask set for efficient predictions:
        # if no autocorrelation output is going to be produced, don't build it.
        tf = getattr(self, "_transform_features", None)
        if tf is not None and len(tf) == len(f_idx) * n_channels:
            will_run = False
            for c in range(n_channels):
                base = c * len(f_idx)
                for n, feat in enumerate(f_idx):
                    if feat in (2, 3, 8, 10, 12) and tf[base + n]:
                        will_run = True
                        break
                if will_run:
                    break
            if not will_run:
                return None

        m = X.shape[2]
        nfft = _ac_nfft(m)
        # Scale each series, and then each centred series, by a power of two, as
        # the per-series path does: exact, and it keeps the FFT's products from
        # under- or overflowing on series of extreme scale.
        flat = X.reshape(n_cases * n_channels, m)
        flat = flat * _pow2_scales(np.abs(flat).max(axis=1))[:, np.newaxis]
        centred = flat - _row_means(flat)[:, np.newaxis]
        centred *= _pow2_scales(np.abs(centred).max(axis=1))[:, np.newaxis]
        # Real-FFT formulation: the power spectrum is real, and the forward FFT
        # of a real even sequence is 2*nfft times its inverse FFT, a scale the
        # lag-0 normalisation cancels - so irfft(|rfft|**2) matches the
        # per-series forward-FFT formulation while keeping every temporary real
        # (half the FFT work and half the memory traffic of complex FFTs).
        F = np.fft.rfft(centred, n=2 * nfft, axis=1)
        ac = np.fft.irfft(F.real * F.real + F.imag * F.imag, n=2 * nfft, axis=1)
        # Normalise by lag 0, guarding zero-variance series (all-zero output,
        # matching _autocorrelations_with_tw).
        ac0 = ac[:, :1].copy()
        zero_var = ac0[:, 0] == 0
        ac0[zero_var] = 1.0
        ac /= ac0
        ac[zero_var] = 0.0
        return ac.reshape(n_cases, n_channels, 2 * nfft)

    def _transform_case(
        self,
        X,
        X_given,
        X_outlier,
        f_idx,
        features,
        fft_cache=None,
        ac_tw=None,
        ac_nfft=0,
        ac_cache=None,
    ):
        # X_given is the case before any normalisation, for catch24's mean and
        # standard deviation, and X_outlier the case for the outlier features
        c22 = np.zeros(len(f_idx) * len(X))

        if hasattr(self, "_transform_features") and len(
            self._transform_features
        ) == len(c22):
            transform_feature = self._transform_features
        else:
            transform_feature = [True] * len(c22)

        f_count = -1
        for i, series in enumerate(X):
            dim = i * len(f_idx)
            smin = numba_min(series)
            smax = numba_max(series)
            # the catch22 features cannot be computed on a constant series
            is_constant = smin == smax
            fft = None
            ac = None

            for n, feature in enumerate(f_idx):
                f_count += 1
                if not transform_feature[f_count]:
                    continue
                if is_constant and feature < 22:
                    c22[dim + n] = np.nan
                    continue

                args = [series]

                if feature == 0 or feature == 1 or feature == 4:
                    args = [series, smin, smax]
                elif feature == 7:
                    args = [series, mean(series)]
                elif feature == 13 or feature == 14:
                    args = [X_outlier[i]]
                elif feature == 15 or feature == 20:
                    if fft is None:
                        if fft_cache is not None:
                            fft = fft_cache[i]
                        else:
                            fft = np.fft.fft(
                                series - mean(series), n=_nextpow2(len(series))
                            )
                    args = [series, fft]
                elif feature in (2, 3, 8, 10, 12):
                    if ac is None:
                        if ac_cache is not None:
                            ac = ac_cache[i]
                        elif ac_tw is not None:
                            ac = _autocorrelations_with_tw(series, ac_tw, ac_nfft)
                        else:
                            ac = _compute_autocorrelations(series)
                    if feature == 2 or feature == 3:
                        args = [ac, len(series)]
                    elif feature == 12:
                        tw = ac_tw if ac_tw is not None else np.empty(0, np.complex128)
                        args = [series, ac, tw, ac_nfft]
                    elif feature == 8:
                        args = [series, ac, _NO_FILTERS, _NO_FILTER_START]
                    else:
                        args = [series, ac]
                if feature == 22:
                    c22[dim + n] = mean(X_given[i])
                elif feature == 23:
                    c22[dim + n] = np.std(X_given[i], ddof=1)
                else:
                    c22[dim + n] = features[feature](*args)

        return c22

    def _spectrum_fft_cache(self, X, f_idx, n_cases):
        """Batch the power-spectrum features' FFT across all cases when it will be used.

        Returns an array of shape (n_cases, n_channels, nfft) with the FFT of each
        mean-centred series, or None if the spectral features (15/20) are absent, the
        input is not an equal-length 3D array, or attribute-skipping means neither
        spectral feature will be computed. Matches the per-case computation to within
        rounding.
        """
        if (15 not in f_idx and 20 not in f_idx) or not (
            isinstance(X, np.ndarray) and X.ndim == 3
        ):
            return None

        n_channels = X.shape[1]

        # Respect the transform_features skip mask set for efficient predictions:
        # if no spectral output is going to be produced, don't build the cache.
        tf = getattr(self, "_transform_features", None)
        if tf is not None and len(tf) == len(f_idx) * n_channels:
            will_run = False
            for c in range(n_channels):
                base = c * len(f_idx)
                for n, feat in enumerate(f_idx):
                    if (feat == 15 or feat == 20) and tf[base + n]:
                        will_run = True
                        break
                if will_run:
                    break
            if not will_run:
                return None

        m = X.shape[2]
        nfft = _nextpow2(m)
        flat = X.reshape(n_cases * n_channels, m)
        # the series means in one compiled call (a call per series from Python
        # costs more than the mean on short series)
        fft = np.fft.fft(flat - _row_means(flat)[:, np.newaxis], n=nfft, axis=1)
        return fft.reshape(n_cases, n_channels, nfft)

    @property
    def get_features_arguments(self):
        """Return feature names for the estimators features argument."""
        return (
            self.features
            if self.features != "all"
            else (
                feature_names + ["Mean", "StandardDeviation"]
                if self.catch24
                else feature_names
            )
        )

    # The feature kernels below follow catch22 v1.2.0's C code. Loops over
    # offset windows index slices of the series, so that numba can vectorise them.

    @staticmethod
    def _DN_HistogramMode_5(X, smin, smax):
        # Mean centre of the most populated of 5 equal-width bins.
        return _histogram_mode(X, 5, smin, smax)

    @staticmethod
    def _DN_HistogramMode_10(X, smin, smax):
        # Mean centre of the most populated of 10 equal-width bins.
        return _histogram_mode(X, 10, smin, smax)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _SB_BinaryStats_diff_longstretch0(X):
        # Longest run of successive non-increasing steps.
        if len(X) < 2:
            return np.nan
        diff_binary = np.zeros(len(X) - 1)
        for i in range(len(X) - 1):
            if X[i + 1] - X[i] > 0:
                diff_binary[i] = 1

        return _long_stretch(diff_binary, 0)

    @staticmethod
    def _DN_OutlierInclude_p_001_mdrmd(X):
        # Time intervals between successive extreme events above the mean.
        return _outlier_include(X)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _DN_OutlierInclude_n_001_mdrmd(X):
        # Time intervals between successive extreme events below the mean.
        return _outlier_include(-X)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _CO_FirstCrossing_acf_1e_pointOfCrossing(X_ac, size):
        # First 1/e crossing of the autocorrelation function, interpolated.
        return _first_crossing(X_ac, size, 0.36787944117144233)  # 1 / e

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _CO_FirstMin_ac(X_ac, size):
        # First minimum of the autocorrelation function.
        if size < 3:
            return np.nan
        for i in range(1, size - 1):
            if X_ac[i] < X_ac[i - 1] and X_ac[i] < X_ac[i + 1]:
                return i
        return size

    @staticmethod
    def _SP_Summaries_fft_area_5_1(X, X_fft):
        # Power in the lowest fifth of the frequencies of the power spectrum.
        return _summaries_fft(X, False, X_fft)

    @staticmethod
    def _SP_Summaries_fft_wq_50(X, X_fft):
        # Median frequency of the power spectrum.
        return _summaries_fft(X, True, X_fft)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _FC_LocalSimple_mean3_stde(X):
        # Standard deviation of the errors of a rolling 3-sample mean forecast.
        if len(X) <= 4:
            return np.nan
        res = _local_simple_mean(X, 3)
        return _stddev(res, len(res))

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _CO_trev_1_num(X):
        # Time-reversibility statistic, ((x_{t+1} - x_t)^3)_t.
        if len(X) < 2:
            return np.nan
        total = 0.0
        for i in range(len(X) - 1):
            total += (X[i + 1] - X[i]) ** 3
        return total / (len(X) - 1)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _CO_HistogramAMI_even_5bin_ami2(X, smin, smax):
        # Automutual information at lag 2 from 5 equal-width bins, with the
        # Panzeri-Treves bias correction.
        if len(X) <= 2:
            return np.nan
        edges = np.zeros(6)
        nb = _hist_edges(smin, smax, 5, edges)
        n_samples = len(X) - 2

        pij = np.zeros((5, 5))
        for t in range(n_samples):
            b1 = _hist_bin_index(X[t], edges, nb)
            b2 = _hist_bin_index(X[t + 2], edges, nb)
            if b1 < 0 or b2 < 0:
                return np.nan
            pij[b1, b2] += 1

        # column-major sums, as MATLAB
        total = 0.0
        for j in range(nb):
            for i in range(nb):
                total += pij[i, j]
        for i in range(nb):
            for j in range(nb):
                pij[i, j] /= total
        pi = np.zeros(5)
        pj = np.zeros(5)
        for i in range(nb):
            for j in range(nb):
                pi[i] += pij[i, j]
        for j in range(nb):
            for i in range(nb):
                pj[j] += pij[i, j]

        ami = 0.0
        m_xy = 0
        for j in range(nb):
            for i in range(nb):
                if pij[i, j] > 0:
                    ami += pij[i, j] * np.log(pij[i, j] / pi[i] / pj[j])
                    m_xy += 1
        m_x = 0
        m_y = 0
        for i in range(nb):
            if pi[i] > 0:
                m_x += 1
            if pj[i] > 0:
                m_y += 1
        return ami - (m_xy - m_x - m_y + 1) / (2.0 * n_samples)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _IN_AutoMutualInfoStats_40_gaussian_fmmi(X):
        # First minimum of the automutual information function.
        tau = min(40, (len(X) + 1) // 2)
        # fewer than 3 lags have no local minimum
        if tau < 3:
            return float(tau)

        ami_prev = _gaussian_ami(X, 1)
        ami_curr = _gaussian_ami(X, 2)
        for i in range(1, tau - 1):
            ami_next = _gaussian_ami(X, i + 2)
            # ami_curr holds lag i + 1
            if ami_curr < ami_prev and ami_curr < ami_next:
                return float(i + 1)
            ami_prev = ami_curr
            ami_curr = ami_next
        return float(tau)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _MD_pNN_pnn40(X):
        # Proportion of successive differences exceeding 0.04 sigma (Mietus 2002).
        if len(X) < 2:
            return np.nan
        nsum = 0
        for i in range(len(X) - 1):
            if np.abs(X[i + 1] - X[i]) * 1000 > 40:
                nsum += 1

        return nsum / (len(X) - 1)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _SB_BinaryStats_mean_longstretch1(X, smean):
        # Longest period of consecutive values above the mean.
        mean_binary = np.zeros(len(X))
        for i in range(len(mean_binary)):
            if X[i] - smean > 0:
                mean_binary[i] = 1

        return _long_stretch(mean_binary, 1)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _SB_MotifThree_quantile_hh(X):
        # Entropy of successive pairs of letters of a 3-letter equiprobable
        # alphabet, with the Miller-Madow bias correction.
        alphabet_size = 3
        n = len(X)
        if n < 2:
            return 0.0
        yt = np.zeros(n, dtype=np.int32)
        _sb_coarsegrain(X, 3, yt)

        counts = np.zeros((alphabet_size, alphabet_size), dtype=np.float64)
        for p in range(n - 1):
            if yt[p] > 0 and yt[p + 1] > 0:
                counts[yt[p] - 1][yt[p + 1] - 1] += 1.0

        hh = 0.0
        non_empty = 0
        for i in range(alphabet_size):
            f = 0.0
            for j in range(alphabet_size):
                p = counts[i][j] / (np.float64(n) - 1.0)
                if p > 0:
                    non_empty += 1
                    f += p * np.log(p)
            hh += -1 * f
        # degrees of freedom constrained by the (fixed, equiprobable) marginals
        df = non_empty - 1 - 2 * (alphabet_size - 1)
        if df > 0:
            hh += df / (2.0 * (n - 1))
        return hh

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _FC_LocalSimple_mean1_taurat(X, X_ac, ac_tw, ac_nfft):
        # Ratio of the (interpolated) first zero crossings of the autocorrelation
        # of the residuals of a 1-sample mean forecast and of the series.
        if len(X) <= 1:
            return np.nan
        y_cross = _first_crossing(X_ac, len(X), 0.0)
        res_cross = _residual_zero_crossing(_local_simple_mean(X, 1), ac_tw, ac_nfft)
        if y_cross == 0 or np.isnan(y_cross) or np.isnan(res_cross):
            return np.nan
        return res_cross / y_cross

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _CO_Embed2_Dist_tau_d_expfit_meandiff(X, X_ac):
        # Mean absolute difference between the distribution of successive
        # distances in the 2-d embedding at tau and an exponential fitted to them.
        size = len(X)
        # first zero of the autocorrelation function, at most size / 10
        tau = 0
        while X_ac[tau] > 0 and tau < size:
            tau += 1
        if tau > size / 10:
            tau = int(np.floor(size / 10))
        n = size - tau - 1
        if n < 1:
            return np.nan

        lo = X[: size - 1]
        hi = X[1:]
        diff = np.zeros(size - 1)
        for i in range(size - 1):
            diff[i] = hi[i] - lo[i]
        a = diff[:n]
        b = diff[tau : tau + n]
        d = np.zeros(n)
        for i in range(n):
            d[i] = np.sqrt(a[i] * a[i] + b[i] * b[i])
        dmin = d[0]
        dmax = d[0]
        for i in range(1, n):
            if d[i] < dmin:
                dmin = d[i]
            if d[i] > dmax:
                dmax = d[i]
        # all distances equal (to within rounding): no distribution to fit
        if dmax - dmin <= 1e-12 * dmax:
            return np.nan
        d_mean = 0.0
        for i in range(n):
            d_mean += d[i]
        d_mean /= n

        num_bins = _hist_num_bins_auto(d, dmin, dmax)
        edges = np.zeros(num_bins + 1)
        _hist_edges(dmin, dmax, num_bins, edges)
        counts = np.zeros(num_bins, dtype=np.int64)
        for i in range(n):
            b = _hist_bin_index(d[i], edges, num_bins)
            if b >= 0:
                counts[b] += 1

        sum_abs = 0.0
        for i in range(num_bins):
            pdf = counts[i] / (n * (edges[i + 1] - edges[i]))
            centre = (edges[i] + edges[i + 1]) / 2
            expf = (1 / d_mean) * np.exp(-centre / d_mean)
            sum_abs += np.abs(pdf - expf)
        return sum_abs / num_bins

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(X):
        # Proportion of timescales in the first of two scaling regimes of
        # detrended fluctuation analysis on every second sample.
        return _fluct_anal_prop_r1(X, 2, True)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(X):
        # Proportion of timescales in the first of two scaling regimes of the
        # range of linearly detrended windows.
        return _fluct_anal_prop_r1(X, 1, False)

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _SB_TransitionMatrix_quantile_3_ac_sumdiagcov(X, X_ac, filters, filter_start):
        # Trace of the covariance of the transition matrix between symbols of a
        # 3-letter alphabet, after anti-aliased downsampling by tau. filters holds
        # the low-pass filter for tau from filter_start[tau] when that is not -1
        # (see _transition_filters); otherwise the filter is built here.
        size = len(X)
        tau = _transition_tau(X_ac, size)

        if tau > 1:
            # Resample the mean-centred series: the filter assumes zeros beyond
            # the ends, which catch22 applies to z-normalised series. Centring
            # keeps the feature unaffected by an offset on raw input.
            smean = 0.0
            for i in range(size):
                smean += X[i]
            smean /= size
            centred = np.zeros(size)
            for i in range(size):
                centred[i] = X[i] - smean
            if tau < len(filter_start) and filter_start[tau] >= 0:
                start = filter_start[tau]
                h = filters[start : start + 20 * tau + 1]
            else:
                h = _lowpass_filter(tau)
            ds = _resample_down(centred, tau, h)
        else:
            ds = np.zeros(size)
            for i in range(size):
                ds[i] = X[i]
        n_ds = len(ds)
        if n_ds < 2:
            return np.nan

        yCG = np.zeros(n_ds, dtype=np.int32)
        _sb_coarsegrain(ds, 3, yCG)
        T = np.zeros((3, 3), dtype=np.float64)
        for i in range(n_ds - 1):
            T[yCG[i] - 1][yCG[i + 1] - 1] += 1
        for i in range(3):
            for j in range(3):
                T[i][j] /= n_ds - 1

        # sum of the variances of the columns of T
        sum_of_diagonal_cov = 0.0
        for j in range(3):
            col_mean = (T[0][j] + T[1][j] + T[2][j]) / 3
            cov = 0.0
            for i in range(3):
                cov += (T[i][j] - col_mean) * (T[i][j] - col_mean)
            sum_of_diagonal_cov += cov / 2
        return sum_of_diagonal_cov

    @staticmethod
    @njit(fastmath=_FASTMATH, cache=True)
    def _PD_PeriodicityWang_th0_01(X):
        # Periodicity measure of (Wang et al. 2007).
        # fewer than 3 lags of the autocorrelation have no peak (and the spline
        # needs distinct breaks)
        if len(X) < 3:
            return 0
        y_spline = _spline_fit(X)

        y_sub = np.zeros(len(X))
        for i in range(len(X)):
            y_sub[i] = X[i] - y_spline[i]

        acmax = int(np.ceil(len(X) / 3.0))
        acf = np.zeros(acmax)
        for tau in range(1, acmax + 1):
            covariance = 0
            for i in range(len(X) - tau):
                covariance += y_sub[i] * y_sub[i + tau]
            acf[tau - 1] = covariance / (len(X) - tau)

        troughs = np.zeros(acmax, dtype=np.int32)
        peaks = np.zeros(acmax, dtype=np.int32)
        n_troughs = 0
        n_peaks = 0
        for i in range(1, acmax - 1):
            slope_in = acf[i] - acf[i - 1]
            slope_out = acf[i + 1] - acf[i]

            if slope_in < 0 and slope_out > 0:
                troughs[n_troughs] = i
                n_troughs += 1
            elif slope_in > 0 and slope_out < 0:
                peaks[n_peaks] = i
                n_peaks += 1

        out = 0
        for i in range(n_peaks):
            j = -1
            while troughs[j + 1] < peaks[i] and j + 1 < n_troughs:
                j += 1

            if j == -1 or acf[peaks[i]] - acf[troughs[j]] < 0.01 or acf[peaks[i]] < 0:
                continue

            # acf[peaks[i]] holds lag peaks[i] + 1
            out = peaks[i] + 1
            break

        return out


@njit(fastmath=_FASTMATH, cache=True)
def _hist_edges(smin, smax, num_bins, edges):
    # num_bins + 1 equal-width bin edges spanning [smin, smax], as hctsa's
    # BF_HistEdges: interior edges lowered, and end edges widened, by 1e-6 of a
    # bin width. Data constant to within rounding get one bin of unit width.
    # Returns the number of bins.
    if smax - smin <= 1e-12 * max(np.abs(smin), np.abs(smax)):
        edges[0] = smin - 0.5
        edges[1] = smin + 0.5
        return 1
    bin_width = (smax - smin) / num_bins
    tol = 1e-6 * bin_width
    for i in range(num_bins + 1):
        edges[i] = (smin + i * bin_width) - tol
    edges[0] = smin - tol
    edges[num_bins] = smax + tol
    return num_bins


@njit(fastmath=_FASTMATH, cache=True)
def _hist_bin_index(x, edges, num_bins):
    # Bin containing x with MATLAB histcounts' convention: [edges[i], edges[i+1]),
    # the last bin also including its right edge. -1 if x is outside the edges.
    if x < edges[0] or x > edges[num_bins]:
        return -1
    if x == edges[num_bins]:
        return num_bins - 1
    # The bin from the spacing of the edges (num_bins + 2e-6 bin widths from the
    # first to the last, see _hist_edges): to within rounding the bin the edges
    # give, which the loops below correct it to.
    i = int((x - edges[0]) / (edges[num_bins] - edges[0]) * (num_bins + 2e-6))
    if i > num_bins - 1:
        i = num_bins - 1
    while i > 0 and x < edges[i]:
        i -= 1
    while i < num_bins - 1 and x >= edges[i + 1]:
        i += 1
    return i


@njit(fastmath=_FASTMATH, cache=True)
def _iqr_matlab(X):
    # Interquartile range with MATLAB's percentile convention.
    s = np.sort(X)
    size = len(s)
    q = np.zeros(2)
    for k in range(2):
        r = size * (0.25 if k == 0 else 0.75) - 0.5
        if r <= 0:
            q[k] = s[0]
        elif r >= size - 1:
            q[k] = s[size - 1]
        else:
            i = int(np.floor(r))
            frac = r - i
            q[k] = s[i] if frac == 0 else s[i] + frac * (s[i + 1] - s[i])
    return q[1] - q[0]


@njit(fastmath=_FASTMATH, cache=True)
def _hist_num_bins_auto(X, smin, smax):
    # hctsa's 'auto' rule: the larger of the Sturges and Freedman-Diaconis numbers
    # of bins (Sturges when the IQR is 0), between 1 and the number of values.
    size = len(X)
    srange = smax - smin
    max_abs = max(np.abs(smax), np.abs(smin))
    num_sturges = np.ceil(np.log2(float(size)) + 1)
    fd_width = 0.0
    q = _iqr_matlab(X)
    if srange > 0 and q > 1e-12 * max_abs:
        fd_width = 2 * q * np.power(float(size), -1.0 / 3.0)
    num_fd = np.ceil(srange / fd_width) if fd_width > 0 else num_sturges
    num_bins = max(num_sturges, num_fd)
    num_bins = max(1.0, min(num_bins, float(size)))
    return int(num_bins)


@njit(fastmath=_FASTMATH, cache=True)
def _histogram_mode(X, num_bins, smin, smax):
    edges = np.zeros(num_bins + 1, dtype=np.float64)
    nb = _hist_edges(smin, smax, num_bins, edges)
    histogram = np.zeros(nb, dtype=np.int64)
    for val in X:
        idx = _hist_bin_index(val, edges, nb)
        if idx >= 0:
            histogram[idx] += 1

    max_count = -1
    for i in range(nb):
        if histogram[i] > max_count:
            max_count = histogram[i]
    max_sum = 0.0
    num_maxs = 0
    for i in range(nb):
        if histogram[i] == max_count:
            max_sum += (edges[i] + edges[i + 1]) / 2
            num_maxs += 1
    return max_sum / num_maxs


@njit(fastmath=_FASTMATH, cache=True)
def _long_stretch(X_binary, val):
    # length of the longest run of consecutive entries equal to val
    max_stretch = 0
    stretch = 0
    for i in range(len(X_binary)):
        if X_binary[i] == val:
            stretch += 1
            if stretch > max_stretch:
                max_stretch = stretch
        else:
            stretch = 0

    return max_stretch


@njit(fastmath=_FASTMATH, cache=True)
def _bit_update(bit, i, n):
    # Fenwick tree point increment at 1-indexed position i (size n).
    while i <= n:
        bit[i] += 1
        i += i & (-i)


@njit(fastmath=_FASTMATH, cache=True)
def _bit_select(bit, k, n, top):
    # Return the 1-indexed position of the k-th smallest present element; top is
    # the largest power of two not above n.
    pos = 0
    pw = top
    while pw > 0:
        nxt = pos + pw
        if nxt <= n and bit[nxt] < k:
            pos = nxt
            k -= bit[nxt]
        pw >>= 1
    return pos + 1


@njit(fastmath=_FASTMATH, cache=True)
def _outlier_include(X):
    # Median, over increasing thresholds, of the median time (relative to the
    # series midpoint) of values above the threshold, as hctsa's
    # DN_OutlierInclude.mdrmd on the (positive) values of X.
    n = len(X)
    inc = 0.01
    tot = 0
    max_val = X[0]
    for v in X:
        if v >= 0:
            tot += 1
        if v > max_val:
            max_val = v
    if max_val < inc:
        return 0.0
    # a z-normalised series has max_val < sqrt(n); larger input would
    # need more thresholds than can be counted
    if max_val > np.sqrt(n):
        return np.nan

    num_thresholds = int(max_val / inc + 1)

    # Bucket each position by the highest threshold index j it satisfies
    # (X[pos] >= j * inc), chaining positions that share one via nxt.
    head = np.full(num_thresholds, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    cnt = np.zeros(num_thresholds, dtype=np.int64)
    for pos in range(n):
        v = X[pos]
        q = v / inc
        if q >= num_thresholds - 1:
            j = num_thresholds - 1
        elif q < 0:
            j = -1
        else:
            j = int(q)
        while j + 1 < num_thresholds and v >= (j + 1) * inc:
            j += 1
        while j >= 0 and v < j * inc:
            j -= 1
        if j >= 0:
            nxt[pos] = head[j]
            head[j] = pos
            cnt[j] += 1

    # counts above each threshold, and from them the thresholds to keep: up to the
    # last with more than 2% of the values, and before the first with one value
    counts = np.zeros(num_thresholds, dtype=np.int64)
    run = 0
    for j in range(num_thresholds - 1, -1, -1):
        run += cnt[j]
        counts[j] = run
    mj = 0
    for j in range(num_thresholds):
        if (counts[j] - 1) * 100.0 / tot > 2:
            mj = j
    fbi = num_thresholds - 1
    for j in range(num_thresholds - 1, -1, -1):
        if counts[j] - 1 == 0:
            fbi = j
    trim_limit = min(mj, fbi)

    top = 1
    while (top << 1) <= n:
        top <<= 1
    bit = np.zeros(n + 1, dtype=np.int64)

    # Sweep thresholds from high to low, inserting positions as they enter the
    # set, and read each median from the Fenwick tree.
    medians = np.zeros(trim_limit + 1)
    m = 0
    denom = n / 2
    for j in range(num_thresholds - 1, -1, -1):
        pos = head[j]
        while pos != -1:
            _bit_update(bit, pos + 1, n)
            m += 1
            pos = nxt[pos]
        if j > trim_limit:
            continue
        if m % 2 == 1:
            med = float(_bit_select(bit, m // 2 + 1, n, top))
        else:
            lo = float(_bit_select(bit, m // 2, n, top))
            hi = float(_bit_select(bit, m // 2 + 1, n, top))
            med = (hi + lo) / 2.0
        medians[j] = med / denom - 1

    return np.median(medians)


@njit(fastmath=_FASTMATH, cache=True)
def _row_means(X):
    # the mean of each row of a 2D array, with aeon's numba mean()
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        out[i] = mean(X[i])
    return out


def _autocorr(X, X_fft):
    ca = np.fft.ifft(_multiply_complex_arr(X_fft))
    return _get_acf(X, ca)


@njit(fastmath=_FASTMATH, cache=True)
def _multiply_complex_arr(X_fft):
    c = np.zeros(len(X_fft), dtype=np.complex128)
    for i, n in enumerate(X_fft):
        c[i] = n * (n.real + 1j * -n.imag)
    return c


@njit(fastmath=_FASTMATH, cache=True)
def _get_acf(X, ca):
    acf = np.zeros(len(X))
    if ca[0].real != 0:
        for i in range(len(X)):
            acf[i] = ca[i].real / ca[0].real
    return acf


@njit(fastmath=_FASTMATH, cache=True)
def _summaries_fft(X, median_freq, X_fft):
    # Single-sided power spectrum of the zero-padded series, as hctsa's
    # SP_Summaries(y,'fft'), with the DC bin dropped and a floor of 1e-12 of the
    # maximum: the median frequency, or the power in the lowest fifth of the
    # frequencies.
    size = len(X)
    nfft = len(X_fft)
    n_bins = nfft // 2
    if n_bins < 2:
        return np.nan
    pi = 3.14159265358979323846
    S = np.zeros(n_bins)
    for k in range(n_bins):
        a = _complex_magnitude(X_fft[k + 1])
        S[k] = 2 * a * a / size / (2 * pi)
    s_max = S[0]
    for k in range(1, n_bins):
        if S[k] > s_max:
            s_max = S[k]
    for k in range(n_bins):
        if S[k] < 1e-12 * s_max:
            S[k] = 1e-12 * s_max

    if median_freq:
        total = 0.0
        for k in range(n_bins):
            total += S[k]
        cs = 0.0
        for k in range(n_bins):
            cs += S[k]
            if cs >= 0.5 * total:
                return 2 * pi * (k + 1) / nfft
        return np.nan

    area = 0.0
    for k in range(n_bins // 5):
        area += S[k]
    return area * (2 * pi * 2 / nfft - 2 * pi * 1 / nfft)


@njit(fastmath=_FASTMATH, cache=True)
def _complex_magnitude(c):
    return np.sqrt(c.real * c.real + c.imag * c.imag)


@njit(fastmath=_FASTMATH, cache=True)
def _local_simple_mean(X, train_length):
    res = np.zeros(len(X) - train_length)
    for i in range(len(res)):
        nsum = 0
        for n in range(train_length):
            nsum += X[i + n]
        res[i] = X[i + train_length] - nsum / train_length
    return res


@njit(fastmath=_FASTMATH, cache=True)
def _first_crossing(X_ac, size, threshold):
    # First crossing of a threshold by the autocorrelation function, linearly
    # interpolated between the lags either side of it; size - 1 if it never
    # crosses.
    if size < 2:
        return np.nan
    for k in range(1, size):
        if X_ac[k] < threshold:
            return (k - 1) + (X_ac[k - 1] - threshold) / (X_ac[k - 1] - X_ac[k])
    return size - 1.0


@njit(fastmath=_FASTMATH, cache=True)
def _residual_zero_crossing(res, ac_tw, ac_nfft):
    # _first_crossing of 0 by the autocorrelation of res. The crossing is almost
    # always at lag 1 (the residuals of a mean forecast are close to differences),
    # so scan the autocovariance directly for the first few lags before falling
    # back to the FFT autocorrelation.
    m = len(res)
    if m < 2:
        return np.nan
    # scale by a power of two (exact) so that the products cannot under- or
    # overflow
    res = res * _pow2_scale(np.max(np.abs(res)))
    res_mean = 0.0
    for i in range(m):
        res_mean += res[i]
    res_mean /= m
    c0 = 0.0
    for i in range(m):
        c0 += (res[i] - res_mean) * (res[i] - res_mean)
    if c0 == 0:
        return np.nan

    prev = 1.0
    for k in range(1, min(m, 16)):
        cov = 0.0
        for i in range(m - k):
            cov += (res[i] - res_mean) * (res[i + k] - res_mean)
        ac_k = cov / c0
        if ac_k < 0:
            return (k - 1) + prev / (prev - ac_k)
        prev = ac_k
    if m <= 16:
        return m - 1.0

    # reuse the precomputed twiddles when the residuals land on the same FFT size
    if len(ac_tw) > 0 and _ac_nfft(m) == ac_nfft:
        ac = _autocorrelations_with_tw(res, ac_tw, ac_nfft)
    else:
        ac = _compute_autocorrelations(res)
    return _first_crossing(ac, m, 0.0)


@njit(fastmath=_FASTMATH, cache=True)
def _gaussian_ami(X, lag):
    # Automutual information at a lag for a Gaussian, -log(1 - r^2) / 2 for the
    # autocorrelation r, floored at 1e-12 inside the log as catch22 does
    n = len(X) - lag
    x = X[:n]
    y = X[lag:]
    mx = 0.0
    my = 0.0
    for j in range(n):
        mx += x[j]
        my += y[j]
    mx /= n
    my /= n
    nom = 0.0
    dx = 0.0
    dy = 0.0
    for j in range(n):
        a = x[j] - mx
        b = y[j] - my
        nom += a * b
        dx += a * a
        dy += b * b
    v = 1e-12
    if dx * dy != 0:
        r = nom / np.sqrt(dx * dy)
        v = max(1.0 - r * r, 1e-12)
    return -0.5 * np.log(v)


@njit(fastmath=_FASTMATH, cache=True)
def _bessel_i0(x):
    # Zeroth-order modified Bessel function of the first kind, from its power
    # series (converges quickly for the small arguments used here)
    half_x2 = (x / 2) * (x / 2)
    term = 1.0
    total = 1.0
    for k in range(1, 500):
        term *= half_x2 / (float(k) * k)
        total += term
        if term < total * 1e-17:
            break
    return total


@njit(fastmath=_FASTMATH, cache=True)
def _transition_tau(X_ac, size):
    # transition_variance's downsampling factor: the first lag with a strictly
    # negative autocorrelation, at most size / 50
    tau = size - 1
    for k in range(1, size):
        if X_ac[k] < 0:
            tau = k
            break
    if tau > size // 50:
        tau = size // 50
    return tau


@njit(fastmath=_FASTMATH, cache=True)
def _lowpass_filter(q):
    # The anti-aliasing filter for downsampling by the integer factor q > 1: a
    # windowed sinc (Kaiser window, beta = 5, half-width 10q) with unit DC gain.
    pi = 3.14159265358979323846
    beta = 5.0
    K = 10 * q
    h = np.zeros(2 * K + 1)
    i0_beta = _bessel_i0(beta)
    h_sum = 0.0
    for m in range(-K, K + 1):
        u = m / q
        g = 1.0 / q if m == 0 else np.sin(pi * u) / (pi * u) / q
        r = m / K
        w = _bessel_i0(beta * np.sqrt(1 - r * r)) / i0_beta
        h[m + K] = g * w
        h_sum += h[m + K]
    for m in range(2 * K + 1):
        h[m] /= h_sum
    return h


@njit(fastmath=_FASTMATH, cache=True)
def _transition_filters(ac, size):
    # The low-pass filters for every downsampling factor that transition_variance
    # uses on a batch of autocorrelations of shape (n_cases, n_channels, lags),
    # built once each instead of once per series. Returns the filters
    # concatenated and, for each factor q, the start of its filter (2 * 10q + 1
    # taps), or -1 for factors no series uses.
    max_q = size // 50
    used = np.zeros(max_q + 1, dtype=np.bool_)
    for i in range(ac.shape[0]):
        for c in range(ac.shape[1]):
            used[_transition_tau(ac[i, c], size)] = True
    start = np.full(max_q + 1, -1, dtype=np.int64)
    total = 0
    for q in range(2, max_q + 1):
        if used[q]:
            start[q] = total
            total += 20 * q + 1
    filters = np.zeros(total)
    for q in range(2, max_q + 1):
        if start[q] >= 0:
            filters[start[q] : start[q] + 20 * q + 1] = _lowpass_filter(q)
    return filters, start


@njit(fastmath=_FASTMATH, cache=True)
def _resample_down(X, q, h):
    # Anti-aliased downsampling by the integer factor q > 1: the low-pass filter
    # h of _lowpass_filter, applied centred on every q-th value with zeros beyond
    # the ends.
    size = len(X)
    K = 10 * q
    M = (size + q - 1) // q

    # h is symmetric, so the convolution is a dot product of X and h over the
    # window
    out = np.zeros(M)
    for i in range(M):
        centre = i * q
        j_lo = max(centre - K, 0)
        j_hi = min(centre + K, size - 1)
        xw = X[j_lo : j_hi + 1]
        hw = h[j_lo - centre + K : j_hi - centre + K + 1]
        acc = 0.0
        for j in range(j_hi - j_lo + 1):
            acc += xw[j] * hw[j]
        out[i] = acc
    return out


@njit(fastmath=_FASTMATH, cache=True)
def _round_half_away(x):
    # C's round(): halves away from zero (np.round rounds them to even)
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)


@njit(fastmath=_FASTMATH, cache=True)
def _line_fit_sse(x, y, start, n):
    # Sum of squared residuals of a least-squares line through n points from start
    sumx = 0.0
    sumx2 = 0.0
    sumxy = 0.0
    sumy = 0.0
    for i in range(start, start + n):
        sumx += x[i]
        sumx2 += x[i] * x[i]
        sumxy += x[i] * y[i]
        sumy += y[i]
    denom = n * sumx2 - sumx * sumx
    m = 0.0
    b = 0.0
    if denom != 0:
        m = (n * sumxy - sumx * sumy) / denom
        b = (sumy * sumx2 - sumx * sumxy) / denom
    sse = 0.0
    for i in range(start, start + n):
        e = m * x[i] + b - y[i]
        sse += e * e
    return sse


@njit(fastmath=_FASTMATH, cache=True)
def _fluct_anal_prop_r1(X, lag, dfa):
    # Fluctuation analysis with a two-regime fit, as hctsa's
    # SC_FluctAnal(x,2,how,50,1,lag,true).prop_r1: the proportion of the
    # (log-spaced) timescales in the first of two scaling regimes. dfa selects
    # detrended fluctuation analysis over the range of linearly detrended windows.
    size = len(X)
    ny = (size + lag - 1) // lag
    if ny < 10:
        return np.nan
    y_cs = np.zeros(ny)
    y_cs[0] = X[0]
    for i in range(1, ny):
        y_cs[i] = y_cs[i - 1] + X[i * lag]

    # up to 50 log-spaced timescales from 5 to floor(ny / 2), rounded and
    # de-duplicated
    tau_step = 50
    lin_low = np.log(5.0)
    lin_high = np.log(float(ny // 2))
    taus = np.zeros(tau_step, dtype=np.int64)
    for i in range(tau_step):
        taus[i] = int(
            _round_half_away(
                np.exp(lin_low + i * (lin_high - lin_low) / (tau_step - 1))
            )
        )
    taus[tau_step - 1] = ny // 2
    taus = np.sort(taus)
    n_tau = 0
    for i in range(tau_step):
        if n_tau == 0 or taus[i] != taus[n_tau - 1]:
            taus[n_tau] = taus[i]
            n_tau += 1
    if n_tau < 8:
        return np.nan

    # fluctuation at each timescale, over the floor(ny / t) full windows
    log_tt = np.zeros(n_tau)
    log_ff = np.zeros(n_tau)
    for idx in range(n_tau):
        t = taus[idx]
        n_windows = ny // t
        sumx = 0.0
        sumx2 = 0.0
        for k in range(t):
            sumx += k + 1.0
            sumx2 += (k + 1.0) * (k + 1.0)
        denom = t * sumx2 - sumx * sumx
        f = 0.0
        for j in range(n_windows):
            w = y_cs[j * t : (j + 1) * t]
            sumxy = 0.0
            sumy = 0.0
            for k in range(t):
                sumxy += (k + 1.0) * w[k]
                sumy += w[k]
            m = 0.0
            b = 0.0
            if denom != 0:
                m = (t * sumxy - sumx * sumy) / denom
                b = (sumy * sumx2 - sumx * sumxy) / denom
            if dfa:
                for k in range(t):
                    resid = w[k] - (m * (k + 1) + b)
                    f += resid * resid
            else:
                rmax = w[0] - (m * 1 + b)
                rmin = rmax
                for k in range(1, t):
                    resid = w[k] - (m * (k + 1) + b)
                    if resid > rmax:
                        rmax = resid
                    if resid < rmin:
                        rmin = resid
                f += (rmax - rmin) * (rmax - rmin)
        if dfa:
            f = np.sqrt(f / (n_windows * t))
        else:
            f = np.sqrt(f / n_windows)
        log_tt[idx] = np.log(float(t))
        log_ff[idx] = np.log(f)

    # two-regime fit: the split minimising the total squared error of the two
    # line fits (which share the split point), each with at least min_points
    min_points = int(max(8.0, _round_half_away(0.25 * n_tau)))
    if n_tau < 2 * min_points:
        return np.nan
    n_splits = n_tau - 2 * min_points + 1
    err = np.zeros(n_splits)
    min_err = np.inf
    for s in range(n_splits):
        i = min_points + s
        sse = _line_fit_sse(log_tt, log_ff, 0, i) + _line_fit_sse(
            log_tt, log_ff, i - 1, n_tau - i + 1
        )
        err[s] = sse / n_tau
        if err[s] < min_err:
            min_err = err[s]
    # the first split within a relative 1e-9 of the best, so that splits tied up
    # to rounding resolve the same way on every platform
    break_pt = min_points
    for s in range(n_splits):
        if err[s] <= min_err * (1 + 1e-9):
            break_pt = min_points + s
            break
    return break_pt / n_tau


@njit(fastmath=_FASTMATH, cache=True)
def _spline_fit(X):
    breaks = np.array([0, len(X) / 2 - 1, len(X) - 1])
    h0 = np.array([breaks[1] - breaks[0], breaks[2] - breaks[1]])
    h_copy = np.array([h0[0], h0[1], h0[0], h0[1]])
    hl = np.array([h_copy[3], h_copy[2], h_copy[1]])
    hr = np.array([h_copy[0], h_copy[1], h_copy[2]])

    hlCS = np.zeros(3)
    hlCS[0] = hl[0]
    for i in range(1, 3):
        hlCS[i] = hlCS[i - 1] + hl[i]

    bl = np.zeros(3)
    for i in range(3):
        bl[i] = breaks[0] - hlCS[i]

    hrCS = np.zeros(3)
    hrCS[0] = hr[0]
    for i in range(1, 3):
        hrCS[i] = hrCS[i - 1] + hr[i]

    br = np.zeros(3)
    for i in range(3):
        br[i] = breaks[2] - hrCS[i]

    breaksExt = np.zeros(9)
    for i in range(3):
        breaksExt[i] = bl[2 - i]
        breaksExt[i + 3] = breaks[i]
        breaksExt[i + 6] = br[i]

    hExt = np.zeros(8)
    for i in range(8):
        hExt[i] = breaksExt[i + 1] - breaksExt[i]

    coeffs = np.zeros((32, 4))
    for i in range(0, 32, 4):
        coeffs[i][0] = 1

    ii = np.zeros((4, 8), dtype=np.int32)
    for i in range(8):
        ii[0][i] = i
        ii[1][i] = min(1 + i, 7)
        ii[2][i] = min(2 + i, 7)
        ii[3][i] = min(3 + i, 7)

    H = np.zeros(32)
    for i in range(32):
        H[i] = hExt[ii[i % 4][int(i / 4)]]

    for k in range(1, 4):
        for j in range(k):
            for u in range(32):
                coeffs[u][j] *= H[u] / (k - j)

        Q = np.zeros((4, 8))
        for u in range(32):
            for m in range(4):
                Q[u % 4][int(u / 4)] += coeffs[u][m]

        for u in range(8):
            for m in range(1, 4):
                Q[m][u] += Q[m - 1][u]

        for u in range(32):
            if u % 4 > 0:
                coeffs[u][k] = Q[u % 4 - 1][int(u / 4)]

        fmax = np.zeros(32)
        for i in range(8):
            for j in range(4):
                fmax[i * 4 + j] = Q[3][i]

        for j in range(k + 1):
            for u in range(32):
                coeffs[u][j] /= fmax[u]

        for i in range(29):
            for j in range(k + 1):
                coeffs[i][j] -= coeffs[3 + i][j]

        for i in range(0, 32, 4):
            coeffs[i][k] = 0

    scale = np.ones(32)
    for k in range(3):
        for i in range(32):
            scale[i] /= H[i]

        for i in range(32):
            coeffs[i][3 - (k + 1)] *= scale[i]

    jj = np.zeros((4, 2), dtype=np.int32)
    for i in range(4):
        for j in range(2):
            if i == 0:
                jj[i][j] = 4 * (1 + j)
            else:
                jj[i][j] = 3

    for i in range(1, 4):
        for j in range(2):
            jj[i][j] += jj[i - 1][j]

    coeffs_out = np.zeros((8, 4))
    for i in range(8):
        coeffs_out[i] = coeffs[jj[i % 4][int(i / 4)] - 1]

    xsB = np.zeros(len(X) * 4)
    indexB = np.zeros(len(xsB), dtype=np.int32)
    breakInd = 1
    for i in range(len(X)):
        if i >= breaks[1] and breakInd < 2:
            breakInd += 1

        for j in range(4):
            xsB[i * 4 + j] = i - breaks[breakInd - 1]
            indexB[i * 4 + j] = j + (breakInd - 1) * 4

    vB = np.zeros(len(xsB))
    for i in range(len(xsB)):
        vB[i] = coeffs_out[indexB[i]][0]

    for i in range(1, 4):
        for j in range(len(xsB)):
            vB[j] = vB[j] * xsB[j] + coeffs_out[indexB[j]][i]

    A = np.zeros(len(X) * 5)
    breakInd = 0
    for i in range(len(xsB)):
        if i / 4 >= breaks[1]:
            breakInd = 1
        A[i % 4 + breakInd + int(i / 4) * 5] = vB[i]

    AT = np.zeros(len(A))
    ATA = np.zeros(25)
    ATb = np.zeros(5)
    for i in range(len(X)):
        for j in range(5):
            AT[j * len(X) + i] = A[i * 5 + j]

    for i in range(5):
        for j in range(5):
            for k in range(len(X)):
                ATA[i * 5 + j] += AT[i * len(X) + k] * A[k * 5 + j]

    for i in range(5):
        for k in range(len(X)):
            ATb[i] += AT[i * len(X) + k] * X[k]

    AElim = np.zeros((5, 5))
    for i in range(5):
        n = i * 5
        AElim[i] = ATA[n : n + 5]

    for i in range(5):
        for j in range(i + 1, 5):
            factor = AElim[j][i] / AElim[i][i]
            ATb[j] = ATb[j] - factor * ATb[i]

            for k in range(i, 5):
                AElim[j][k] = AElim[j][k] - factor * AElim[i][k]

    x = np.zeros(5)
    for i in range(4, -1, -1):
        bMinusATemp = ATb[i]
        for j in range(i + 1, 5):
            bMinusATemp -= x[j] * AElim[i][j]

        x[i] = bMinusATemp / AElim[i][i]

    C = np.zeros((5, 8))
    for i in range(32):
        C[int(i % 4 + int(i / 4) % 2)][int(i / 4)] = coeffs_out[i % 8][int(i / 8)]

    coeffs_spline = np.zeros((2, 4))
    for j in range(8):
        coefc = int(j / 2)
        coefr = j % 2
        for i in range(5):
            coeffs_spline[coefr][coefc] += C[i][j] * x[i]

    y_out = np.zeros(len(X))
    for i in range(len(X)):
        second_half = 0 if i < breaks[1] else 1
        y_out[i] = coeffs_spline[second_half][0]

    for i in range(1, 4):
        for j in range(len(X)):
            second_half = 0 if j < breaks[1] else 1
            y_out[j] = (
                y_out[j] * (j - breaks[1] * second_half) + coeffs_spline[second_half][i]
            )

    return y_out


def _feature_index(name, catch24):
    # index of a feature by its name, short name or catch22 v0.x name
    if name in feature_names_short:
        return feature_names_short.index(name)
    if name in feature_names:
        return feature_names.index(name)
    # TODO remove the catch22 v0.x names in v1.8.0
    if name in legacy_feature_names:
        return legacy_feature_names.index(name)
    if name in _legacy_short_names:
        return _legacy_short_names[name]
    if catch24 and name in _catch24_names:
        return _catch24_names[name]
    raise ValueError("Invalid feature selection.")


# TODO remove with the catch22 v0.x names in v1.8.0
def _legacy_names_in(features):
    # the catch22 v0.x names among the selected features
    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, (list, tuple)):
        return []
    return [
        f
        for f in features
        if isinstance(f, str)
        and f not in feature_names
        and f not in feature_names_short
        and (f in legacy_feature_names or f in _legacy_short_names)
    ]


def _verify_features(features, catch24):
    if isinstance(features, str):
        if features == "all":
            f_idx = [*range(22)]
            if catch24:
                f_idx += [22, 23]
        else:
            f_idx = [_feature_index(features, catch24)]
    elif isinstance(features, int):
        if features >= 0 and features < 22:
            f_idx = [features]
        elif catch24 and features == 22:
            f_idx = [22]
        elif catch24 and features == 23:
            f_idx = [23]
        else:
            raise ValueError("Invalid feature selection.")
    elif isinstance(features, (list, tuple)):
        if len(features) > 0:
            f_idx = []
            for f in features:
                if isinstance(f, str):
                    f_idx.append(_feature_index(f, catch24))
                elif isinstance(f, int):
                    if f >= 0 and f < 22:
                        f_idx.append(f)
                    elif catch24 and f == 22:
                        f_idx.append(22)
                    elif catch24 and f == 23:
                        f_idx.append(23)
                    else:
                        raise ValueError("Invalid feature selection.")
                else:
                    raise ValueError("Invalid feature selection.")
        else:
            raise ValueError("Invalid feature selection.")
    else:
        raise ValueError("Invalid feature selection.")

    return f_idx


def _mask_unestimable(c22_array, X, f_idx):
    """Replace, in place, features that could not be estimated by catch22's values.

    A feature is masked when it is NaN for a series of finite values; a series
    containing NaN or inf keeps NaN features, as in catch22.
    """
    n_feats = len(f_idx)
    if isinstance(X, np.ndarray) and X.ndim == 3:
        # equal length: one mask value per feature, applied a column at a time
        finite = np.isfinite(X).all(axis=2)
        for col in range(c22_array.shape[1]):
            channel, n = divmod(col, n_feats)
            if f_idx[n] >= 22:
                continue
            rows = np.isnan(c22_array[:, col]) & finite[:, channel]
            if rows.any():
                c22_array[rows, col] = _mask_value(f_idx[n], X.shape[2])
        return
    for case, col in np.argwhere(np.isnan(c22_array)):
        channel, n = divmod(col, n_feats)
        feature = f_idx[n]
        if feature >= 22:
            continue
        series = X[case][channel]
        if np.all(np.isfinite(series)):
            c22_array[case, col] = _mask_value(feature, len(series))


def _normalise(X):
    # z-normalise each series of a 3D array or a list of 2D arrays
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return z_normalise_series_3d(X)
    return [z_normalise_series_2d(x) for x in X]


def _pow2_scales(max_abs):
    # _pow2_scale for an array of maxima (frexp gives 0 and inf an exponent of 0)
    e = np.frexp(max_abs)[1]
    return np.ldexp(1.0, np.clip(-e, -1022, 1023))


def _nextpow2(n):
    # smallest power of two not below n
    return 1 << (int(n) - 1).bit_length()


@njit(fastmath=_FASTMATH, cache=True)
def _ac_nfft(n):
    nFFT = int(np.log2(n))
    if 2**nFFT == n:
        nFFT = n * 2
    else:
        nFFT = (2 ** (nFFT + 1)) * 2
    return nFFT


@njit(fastmath=_FASTMATH, cache=True)
def _ac_twiddles(nFFT):
    # FFT twiddle factors; depend only on nFFT so can be reused across the many
    # autocorrelation calls of a transform (all series share a length).
    tw = np.zeros(nFFT * 2, dtype=np.complex128)
    PI = np.pi
    for i in range(nFFT):
        tmp = 0.0 - PI * i / nFFT * 1j
        tw[i] = np.exp(tmp)
    return tw


@njit(fastmath=_FASTMATH, cache=True)
def _autocorrelations_with_tw(X, tw, nFFT):
    # The (normalised) autocorrelation does not depend on the scale of X, so
    # scale the centred series by powers of two, which is exact, to keep the
    # products in the FFT from under- or overflowing.
    s1 = _pow2_scale(np.max(np.abs(X)))
    mean = np.mean(X * s1)
    s2 = _pow2_scale(np.max(np.abs(X * s1 - mean)))
    F = np.zeros(nFFT * 2, dtype=np.complex128)
    for i in range(len(X)):
        F[i] = complex((X[i] * s1 - mean) * s2, 0.0)
    for i in range(len(X), nFFT):
        F[i] = complex(0.0, 0.0)
    F = _fft(F, tw)
    # dot multiply
    F = np.multiply(F, np.conj(F))
    F = _fft(F, tw)
    divisor = F[0]
    if np.real(divisor) == 0 and np.imag(divisor) == 0:
        return np.zeros(nFFT * 2, dtype=np.float64)
    F = F / divisor
    out = np.real(F)
    return out


@njit(fastmath=_FASTMATH, cache=True)
def _compute_autocorrelations(X):
    nFFT = _ac_nfft(len(X))
    return _autocorrelations_with_tw(X, _ac_twiddles(nFFT), nFFT)


@njit(fastmath=_FASTMATH, cache=True)
def _fft(a, tw):
    n = a.shape[0]
    log_n = int(np.log2(n))
    out = np.empty_like(a)

    # Bit-reversed addressing permutation
    for i in range(n):
        j = 0
        for k in range(log_n):
            j = (j << 1) | ((i >> k) & 1)
        out[j] = a[i]

    # Iterative FFT computation
    step = 1
    while step < n:
        halfstep = step
        step = 2 * step
        for i in range(0, n, step):
            for j in range(halfstep):
                t = tw[j * (n // step)] * out[i + j + halfstep]
                u = out[i + j]
                out[i + j] = u + t
                out[i + j + halfstep] = u - t

    return out


@njit(fastmath=_FASTMATH, cache=True)
def _stddev(a, size):
    m = np.mean(a[:size])
    sd = np.sqrt(np.sum((a[:size] - m) ** 2) / (size - 1))
    return sd


@njit(fastmath=_FASTMATH, cache=True)
def _sb_coarsegrain(y, num_groups, labels):
    th = np.zeros((num_groups + 1), dtype=np.float64)
    ls = np.zeros((num_groups + 1), dtype=np.float64)
    # linspace
    step_size = 1 / (num_groups)
    start = 0
    for i in range(num_groups + 1):
        ls[i] = start
        start += step_size
    # Sort once and reuse for every quantile: _quantile would otherwise re-sort
    # the same array num_groups + 1 times.
    tmp = np.sort(y)
    for i in range(num_groups + 1):
        th[i] = _quantile_sorted(tmp, ls[i])
    # the minimum is in the first group (th[0] - 1 == th[0] beyond 2**53)
    th[0] = -np.inf
    # Single pass over y: the group ranges (th[i], th[i+1]] are disjoint, so the
    # first match is the only match (equivalent to the original num_groups passes).
    for j in range(len(y)):
        for i in range(num_groups):
            if y[j] > th[i] and y[j] <= th[i + 1]:
                labels[j] = i + 1
                break


@njit(fastmath=_FASTMATH, cache=True)
def _quantile_sorted(tmp, quant):
    # Linear-interpolation quantile of an already-sorted array.
    n = len(tmp)
    q = 0.5 / n
    if quant < q:
        return tmp[0]
    elif quant > (1 - q):
        return tmp[n - 1]

    quant_idx = n * quant - 0.5
    idx_left = int(np.floor(quant_idx))
    idx_right = int(np.ceil(quant_idx))
    return tmp[idx_left] + (quant_idx - idx_left) * (tmp[idx_right] - tmp[idx_left]) / (
        idx_right - idx_left
    )


@njit(fastmath=_FASTMATH, cache=True)
def _quantile(X, quant):
    return _quantile_sorted(np.sort(X), quant)


# Module-level aliases so the dispatch kernel below can call the feature
# kernels: numba resolves plain module globals at compile time, but cannot look
# up attributes on a Python class. All of these are the same compiled
# dispatchers the Python path calls.
_f1ecac = Catch22._CO_FirstCrossing_acf_1e_pointOfCrossing
_first_min_ac = Catch22._CO_FirstMin_ac
_ami_even_5bin_ami2 = Catch22._CO_HistogramAMI_even_5bin_ami2
_trev_1_num = Catch22._CO_trev_1_num
_pnn40 = Catch22._MD_pNN_pnn40
_binstats_mean_longstretch1 = Catch22._SB_BinaryStats_mean_longstretch1
_transition_matrix_3ac = Catch22._SB_TransitionMatrix_quantile_3_ac_sumdiagcov
_periodicity_wang = Catch22._PD_PeriodicityWang_th0_01
_embed2_dist_expfit = Catch22._CO_Embed2_Dist_tau_d_expfit_meandiff
_ami_stats_40_fmmi = Catch22._IN_AutoMutualInfoStats_40_gaussian_fmmi
_local_simple_mean1_taurat = Catch22._FC_LocalSimple_mean1_taurat
_outlier_include_n = Catch22._DN_OutlierInclude_n_001_mdrmd
_binstats_diff_longstretch0 = Catch22._SB_BinaryStats_diff_longstretch0
_motif_three_quantile_hh = Catch22._SB_MotifThree_quantile_hh
_fluct_anal_rsrange = Catch22._SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
_fluct_anal_dfa = Catch22._SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
_local_simple_mean3_stde = Catch22._FC_LocalSimple_mean3_stde


@njit(fastmath=_FASTMATH, cache=True)
def _zeros_promoted(series, n):
    """Allocate a length-n zero array typed to match catch22's dtype rule.

    Applies numpy's promotion for dividing by one (float32 input stays float32,
    every other dtype -- including integer input -- promotes to float64), so
    that output buffers keep the input's precision instead of silently
    upcasting to float64.
    """
    dtype = (series[:0] / 1).dtype
    return np.zeros(n, dtype=dtype)


@njit(fastmath=_FASTMATH, cache=True)
def _transform_case_numba(
    X,
    X_outlier,
    f_idx,
    keep,
    fft_case,
    ac_case,
    ac_tw,
    ac_nfft,
    means,
    stds,
    tv_filters,
    tv_start,
):
    """Compiled equivalent of Catch22._transform_case for equal-length input.

    Runs the per-(channel, feature) dispatch loop in compiled code, calling the
    same feature kernels the Python loop calls, with the same lazily shared
    intermediates (min/max, mean, FFT row, autocorrelation row). The FFT and
    autocorrelation always come from the batch caches: the caller only routes
    here for 3D input, where a cache is built whenever its features are present
    and unmasked. `X_outlier` is the case the outlier features are computed on
    (z-normalised with ``outlier_norm``, else `X` itself). `means` and `stds`
    carry catch24's mean and np.std per channel, of the series before any
    normalisation, when catch24's features are requested (numba's np.std can
    round differently from numpy's). `tv_filters` and `tv_start` hold
    transition_variance's low-pass filters (see _transition_filters).
    """
    n_feats = len(f_idx)
    c22 = _zeros_promoted(X[0], n_feats * len(X))

    f_count = -1
    for i in range(len(X)):
        series = X[i]
        dim = i * n_feats
        smin = numba_min(series)
        smax = numba_max(series)
        # the catch22 features cannot be computed on a constant series
        is_constant = smin == smax
        # typed placeholders; real values are assigned before first use
        fft = ac_tw[:0]
        have_fft = False
        ac = np.empty(0, np.float64)
        have_ac = False

        for n in range(n_feats):
            f_count += 1
            if not keep[f_count]:
                continue
            feature = f_idx[n]
            if is_constant and feature < 22:
                c22[dim + n] = np.nan
                continue

            if feature == 0 or feature == 1 or feature == 4:
                if feature == 0 or feature == 1:
                    # num_bins from the runtime feature id (0 -> 5, 1 -> 10)
                    c22[dim + n] = _histogram_mode(
                        series, (feature + 1) * 5, smin, smax
                    )
                else:
                    c22[dim + n] = _ami_even_5bin_ami2(series, smin, smax)
            elif (
                feature == 2
                or feature == 3
                or feature == 8
                or feature == 10
                or feature == 12
            ):
                if not have_ac:
                    ac = ac_case[i]
                    have_ac = True
                if feature == 2:
                    c22[dim + n] = _f1ecac(ac, len(series))
                elif feature == 3:
                    c22[dim + n] = _first_min_ac(ac, len(series))
                elif feature == 8:
                    c22[dim + n] = _transition_matrix_3ac(
                        series, ac, tv_filters, tv_start
                    )
                elif feature == 10:
                    c22[dim + n] = _embed2_dist_expfit(series, ac)
                else:
                    c22[dim + n] = _local_simple_mean1_taurat(
                        series, ac, ac_tw, ac_nfft
                    )
            elif feature == 13:
                c22[dim + n] = _outlier_include(X_outlier[i])
            elif feature == 14:
                c22[dim + n] = _outlier_include_n(X_outlier[i])
            elif feature == 15 or feature == 20:
                if not have_fft:
                    fft = fft_case[i]
                    have_fft = True
                c22[dim + n] = _summaries_fft(series, feature == 20, fft)
            elif feature == 7:
                c22[dim + n] = _binstats_mean_longstretch1(series, mean(series))
            elif feature == 22:
                c22[dim + n] = means[i]
            elif feature == 23:
                c22[dim + n] = stds[i]
            elif feature == 5:
                c22[dim + n] = _trev_1_num(series)
            elif feature == 6:
                c22[dim + n] = _pnn40(series)
            elif feature == 9:
                c22[dim + n] = _periodicity_wang(series)
            elif feature == 11:
                c22[dim + n] = _ami_stats_40_fmmi(series)
            elif feature == 16:
                c22[dim + n] = _binstats_diff_longstretch0(series)
            elif feature == 17:
                c22[dim + n] = _motif_three_quantile_hh(series)
            elif feature == 18:
                c22[dim + n] = _fluct_anal_rsrange(series)
            elif feature == 19:
                c22[dim + n] = _fluct_anal_dfa(series)
            elif feature == 21:
                c22[dim + n] = _local_simple_mean3_stde(series)

    return c22
