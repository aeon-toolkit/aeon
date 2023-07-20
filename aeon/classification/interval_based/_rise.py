# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Random Interval Spectral Ensemble (RISE) classifier."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["RandomIntervalSpectralEnsembleClassifier"]


class RandomIntervalSpectralEnsembleClassifier(ClassifierMixin, BaseIntervalForest):
    """Random Interval Spectral Ensemble (RISE) classifier.

    Input: n series length m
    For each tree
        - sample a random intervals
        - take the ACF and PS over this interval, and concatenate features
        - build a tree on new features
    Ensemble the trees through averaging probabilities.

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    min_interval_length : int, float, list, or tuple, default=3
        Minimum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the minimum interval length.

        Different minimum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.
    max_interval_length : int, float, list, or tuple, default=np.inf
        Maximum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the maximum interval length.

        Different maximum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.

        Ignored for supervised interval_selection_method inputs.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    class_dictionary_ : dict
        A dictionary mapping class labels to class indices in classes_.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of TransformerMixin
        Stores the interval extraction transformer for all estimators.
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances_ ,total_intervals * att_subsample_size)
        The transformed dataset for all estimators. Only saved when
        save_transformed_data is true.

    See Also
    --------
    RISERegressor

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/interval_based/RISE.java>`_.

    References
    ----------
    .. [1] Jason Lines, Sarah Taylor and Anthony Bagnall, "Time Series Classification
       with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based
       Ensembles", ACM Transactions on Knowledge and Data Engineering, 12(5): 2018

    Examples
    --------
    >>> from tsml.interval_based import RISEClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=10, series_length=12, random_state=0)
    >>> clf = RISEClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    RISEClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        min_interval_length=3,
        max_interval_length=np.inf,
        acf_lag=100,
        acf_min_values=4,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pyfftw=True,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values

        self.use_pyfftw = use_pyfftw
        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)

        if isinstance(base_estimator, CITClassifier):
            replace_nan = "nan"
        else:
            replace_nan = 0

        interval_features = [
            PeriodogramTransformer(use_pyfftw=True, pad_with="mean"),
            AutocorrelationFunctionTransformer(
                n_lags=acf_lag, min_values=acf_min_values
            ),
        ]

        super(RISEClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=1,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=None,
            att_subsample_size=None,
            replace_nan=replace_nan,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        return self._predict_proba(X)

    @classmethod
    def get_test_params(
        cls, parameter_set: Union[str, None] = None
    ) -> Union[dict, List[dict]]:
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return {
            "n_estimators": 2,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            RandomIntervalSpectralEnsemble provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10}
        else:
            return {
                "n_estimators": 2,
                "acf_lag": 10,
                "min_interval": 5,
            }


@jit(parallel=True, cache=True, nopython=True)
def acf(x, max_lag):
    """Autocorrelation function transform.

    currently calculated using standard stats method. We could use inverse of power
    spectrum, especially given we already have found it, worth testing for speed and
    correctness. HOWEVER, for long series, it may not give much benefit, as we do not
    use that many ACF terms.

    Parameters
    ----------
    x : array-like shape = [interval_width]
    max_lag: int
        The number of ACF terms to find.

    Returns
    -------
    y : array-like shape = [max_lag]
    """
    y = np.empty(max_lag)
    length = len(x)
    for lag in prange(1, max_lag + 1):
        # Do it ourselves to avoid zero variance warnings
        lag_length = length - lag
        x1, x2 = x[:-lag], x[lag:]
        s1 = np.sum(x1)
        s2 = np.sum(x2)
        m1 = s1 / lag_length
        m2 = s2 / lag_length
        ss1 = np.sum(x1 * x1)
        ss2 = np.sum(x2 * x2)
        v1 = ss1 - s1 * m1
        v2 = ss2 - s2 * m2
        v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
        if v1_is_zero and v2_is_zero:  # Both zero variance,
            # so must be 100% correlated
            y[lag - 1] = 1
        elif v1_is_zero or v2_is_zero:  # One zero variance
            # the other not
            y[lag - 1] = 0
        else:
            y[lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)
        # _x = np.vstack((x[:-lag], x[lag:]))
        # s = np.sum(_x, axis=1)
        # ss = np.sum(_x * _x, axis=1)
        # v = ss - s * s / l
        # zero_variances = v <= 1e-9
        # i = lag - 1
        # if np.all(zero_variances):  # Both zero variance,
        #     # so must be 100% correlated
        #     y[i] = 1
        # elif np.any(zero_variances):  # One zero variance
        #     # the other not
        #     y[i] = 0
        # else:
        #     m = _x - s.reshape(2, 1) / l
        #     y[i] = (m[0] @ m[1]) / np.sqrt(np.prod(v))

    return y


#        y[lag - 1] = np.corrcoef(x[lag:], x[:-lag])[0][1]
#        if np.isnan(y[lag - 1]) or np.isinf(y[lag-1]):
#            y[lag-1]=0

# @jit(parallel=True, cache=True, nopython=True)
# def _acf(x, max_lag):
#     y = np.empty(max_lag)
#     length = len(x)
#     n = length - np.arange(1, max_lag + 1)
#     # _x = np.array([x[:-1], x[:0:-1]])
#     # from_end_to_lag = slice(-1, -max_lag - 1, -1)
#     # cs = np.cumsum(_x, axis=1)[:, from_end_to_lag]
#     # cm = cs / n
#     # css = np.cumsum(_x * _x, axis=1)[:, from_end_to_lag]
#     # cv = css - cs
#
#     a, b = x[:-1], x[:0:-1]
#     from_end_to_lag = slice(-1, -max_lag - 1, -1)
#     cs1 = np.cumsum(a)[from_end_to_lag] / n
#     cs2 = np.cumsum(b)[from_end_to_lag] / n
#     css1 = np.cumsum(a * a)[from_end_to_lag] / n
#     css2 = np.cumsum(b * b)[from_end_to_lag] / n
#     cv1 = css1 - cs1 * cs1
#     cv2 = css2 - cs2 * cs2
#     covar = cv1 * cv2
#
#     for lag in prange(1, max_lag + 1):
#         idx = lag - 1
#         m1, m2, l = cs1[idx], cs2[idx], n[idx]
#         y[idx] = np.sum((x[:-lag] - m1) * (x[lag:] - m2)) / l
#     # both_zero = (cv1 <= 1e-9) & (cv2 <= 1e-9)
#     # one_zero = (cv1 <= 1e-9) ^ (cv2 <= 1e-9)
#     cv1_is_zero, cv2_is_zero = cv1 <= 1e-9, cv2 <= 1e-9
#     non_zero = ~cv1_is_zero & ~cv2_is_zero
#     y[cv1_is_zero & cv2_is_zero] = 1  # Both zero variance,
#     # so must be 100% correlated
#     y[cv1_is_zero ^ cv2_is_zero] = 0  # One zero variance
#     # the other not
#     y[non_zero] /= np.sqrt(covar[non_zero])
#
#     return y


# @jit(parallel=True, cache=True, nopython=True)
def matrix_acf(x, num_cases, max_lag):
    """Autocorrelation function transform.

    Calculated using standard stats method. We could use inverse of power
    spectrum, especially given we already have found it, worth testing for speed and
    correctness. HOWEVER, for long series, it may not give much benefit, as we do not
    use that many ACF terms.

    Parameters
    ----------
    x : array-like shape = [num_cases, interval_width]
    max_lag: int
        The number of ACF terms to find.

    Returns
    -------
    y : array-like shape = [num_cases,max_lag]

    """
    y = np.empty(shape=(num_cases, max_lag))
    length = x.shape[1]
    for lag in prange(1, max_lag + 1):
        # Could just do it ourselves ... TO TEST
        #            s1=np.sum(x[:-lag])/x.shape()[0]
        #            ss1=s1*s1
        #            s2=np.sum(x[lag:])
        #            ss2=s2*s2
        #
        lag_length = length - lag
        x1, x2 = x[:, :-lag], x[:, lag:]
        s1 = np.sum(x1, axis=1)
        s2 = np.sum(x2, axis=1)
        m1 = s1 / lag_length
        m2 = s2 / lag_length
        s12 = np.sum(x1 * x2, axis=1)
        ss1 = np.sum(x1 * x1, axis=1)
        ss2 = np.sum(x2 * x2, axis=1)
        v1 = ss1 - s1 * m1
        v2 = ss2 - s2 * m2
        v12 = s12 - s1 * m2
        v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
        non_zero = ~v1_is_zero & ~v2_is_zero
        # y[:, lag - 1] = np.sum((x1 - m1[:, None]) *
        # (x2 - m2[:, None]), axis=1)
        y[v1_is_zero & v2_is_zero, lag - 1] = 1  # Both zero variance,
        # so must be 100% correlated
        y[v1_is_zero ^ v2_is_zero, lag - 1] = 0  # One zero variance
        # the other not
        var = (v1 * v2)[non_zero]
        y[non_zero, lag - 1] = v12[non_zero] / np.sqrt(var)
    #     # y[lag - 1] = np.corrcoef(x[:, lag:], x[:, -lag])[0][1]
    #     # if np.isnan(y[lag - 1]) or np.isinf(y[lag - 1]):
    #     #     y[lag - 1] = 0
    return y


def ps(x, sign=1, n=None, pad="mean"):
    """Power spectrum transformer.

    Power spectrum transform, currently calculated using np function.
    It would be worth looking at ff implementation, see difference in speed
    to java.

    Parameters
    ----------
    x : array-like shape = [interval_width]
    sign : {-1, 1}, default = 1
    n : int, default=None
    pad : str or function, default='mean'
        controls the mode of the pad function
        see numpy.pad for more details
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns
    -------
    y : array-like shape = [len(x)/2]
    """
    x_len = x.shape[-1]
    x_is_1d = x.ndim == 1
    # pad or slice series if length is not of power of 2 or n is specified
    if x_len & (x_len - 1) != 0 or n:
        # round n (or the length of x) to next power of 2
        # when n is not specified
        if not n:
            n = _round_to_nearest_power_of_two(x_len)
        # pad series up to n when n is larger otherwise slice series up to n
        if n > x_len:
            pad_length = (0, n - x_len) if x_is_1d else ((0, 0), (0, n - x_len))
            x_in_power_2 = np.pad(x, pad_length, mode=pad)
        else:
            x_in_power_2 = x[:n] if x_is_1d else x[:, :n]
    else:
        x_in_power_2 = x
    # use sign to determine inverse or normal fft
    # using the norm in numpy fft function
    # backward = normal fft, forward = inverse fft (divide by n after fft)
    # note: use the following code when upgrade numpy to 1.20
    # norm = "backward" if sign > 0 else "forward"
    # fft = np.fft.rfft(x_in_power_2, norm=norm)
    if sign < 0:
        x_in_power_2 /= n
    fft = np.fft.rfft(x_in_power_2)
    fft = fft[:-1] if x_is_1d else fft[:, :-1]
    return np.abs(fft)


@jit("int64(int64)", cache=True, nopython=True)
def _round_to_nearest_power_of_two(n):
    return int64(1 << round(np.log2(n)))
