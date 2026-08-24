"""TDE classifiers and their specialised SFA transform.

Dictionary based TDE classifiers based on SFA transform. Contains a single
IndividualTDE and TDE.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["TemporalDictionaryEnsemble", "IndividualTDE", "histogram_intersection"]

import math
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from numba import njit, types
from numba.typed import Dict
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.utils.validation import check_n_jobs

# largest number of cases for which the LOOCV nearest neighbour search
# materialises the full similarity matrix (n^2 int32); above this the
# per-case search is used instead
_SYMMETRIC_LOOCV_MAX_N = 4096


def _kernel_ridge_preds(x_hist, y_hist, candidates):
    """Kernel ridge predictions for the ensemble parameter selection.

    The same computation as sklearn StandardScaler + KernelRidge(
    kernel="poly", degree=1) with default alpha=1, gamma=1/n_features and
    coef0=1, i.e. linear ridge regression on standardised features in dual
    form, without the per-call sklearn validation overhead.
    """
    mean = x_hist.mean(axis=0)
    std = x_hist.std(axis=0)
    std[std == 0.0] = 1.0
    xs = (x_hist - mean) / std
    cs = (candidates - mean) / std

    gamma = 1.0 / xs.shape[1]
    k = xs @ xs.T * gamma + 1.0
    k.flat[:: k.shape[0] + 1] += 1.0  # alpha = 1 regularisation
    dual = np.linalg.solve(k, y_hist)
    return (cs @ xs.T * gamma + 1.0) @ dual


class TemporalDictionaryEnsemble(BaseClassifier):
    """
    Temporal Dictionary Ensemble (TDE).

    Implementation of the dictionary based Temporal Dictionary Ensemble as described
    in [1]_.

    Overview: Input 'n' series of length 'm' with 'd' channels.
    TDE searches 'k' parameter values, using kernel ridge regression over
    previously evaluated parameter combinations to predict the accuracy of
    candidate parameter sets, and evaluates each selected set with a LOOCV.
    (The reference paper [1] describes this step as a Gaussian process
    regressor.) It then retains 's' ensemble members.
    There are six primary parameters for individual classifiers:
            - alpha: alphabet size
            - w: window length
            - l: word length
            - p: normalise/no normalise
            - h: levels
            - b: MCB/IGB
    For any combination, an individual TDE classifier slides a window of
    length w along the series. The w length window is shortened to
    an l length word through taking a Fourier transform and keeping the
    first l/2 complex coefficients. These coefficients are then discretised
    into alpha possible values, to form a word of length l using breakpoints
    found using b. A histogram of words for each series is formed and stored,
    using a spatial pyramid of h levels. For multivariate series, accuracy
    from a reduced histogram is used to select channels.

    fit involves finding n histograms.
    predict uses 1 nearest neighbour with the histogram intersection
    similarity function.

    Parameters
    ----------
    n_parameter_samples : int, default=250
        Number of parameter combinations to consider for the final ensemble.
    max_ensemble_size : int, default=50
        Maximum number of estimators in the ensemble.
    max_win_len_prop : float, default=1
        Maximum window length as a proportion of series length, must be between 0 and 1.
    min_window : int, default=10
        Minimum window length.
    randomly_selected_params : int, default=50
        Number of parameters randomly selected before the kernel ridge regression
        guided parameter selection is used.
    bigrams : bool or None, default=None
        Whether to use bigrams, defaults to true for univariate data and false for
        multivariate data.
    channel_threshold : float, default=0.85
        Channel accuracy threshold for multivariate data, must be between 0 and 1.
    dim_threshold : float, default="deprecated"
        Deprecated alias for ``channel_threshold``. Will be removed in v1.7.0.
    max_channels : int, default=20
        Max number of channels per classifier for multivariate data.
    max_dims : int, default="deprecated"
        Deprecated alias for ``max_channels``. Will be removed in v1.7.0.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_parameter_samples.
        Default of 0 means n_parameter_samples is used.
    contract_max_n_parameter_samples : int, default=np.inf
        Max number of parameter combinations to consider when time_limit_in_minutes is
        set.
    typed_dict : bool, default="deprecated"
        Has no effect: word counts are now stored as sorted arrays.

        Deprecated and will be removed in v1.7.0.
    train_estimate_method : str, default="loocv"
        Method used to generate train estimates in `fit_predict` and
        `fit_predict_proba`. Options are "loocv" for leave one out cross validation and
        "oob" for out of bag estimates.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `predict`. `fit` is
        single threaded. ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of channels per case.
    n_timepoints_ : int
        The length of each series.
    estimators_ : list of shape (n_estimators) of IndividualTDE
        The collections of estimators trained in fit.
    n_estimators_ : int
        The final number of classifiers used. Will be <= `max_ensemble_size`.
    weights_ : list of shape (n_estimators) of float
        Weight of each estimator in the ensemble.

    See Also
    --------
    IndividualTDE, ContractableBOSS
        Components usable in TDE.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/TDE.java>`_.

    References
    ----------
    .. [1] Matthew Middlehurst, James Large, Gavin Cawley and Anthony Bagnall
        "The Temporal Dictionary Ensemble (TDE) Classifier for Time Series
        Classification", in proceedings of the European Conference on Machine Learning
        and Principles and Practice of Knowledge Discovery in Databases, 2020.

    Examples
    --------
    >>> from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = TemporalDictionaryEnsemble(
    ...     n_parameter_samples=10,
    ...     max_ensemble_size=3,
    ...     randomly_selected_params=5,
    ... )
    >>> clf.fit(X_train, y_train)
    TemporalDictionaryEnsemble(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "dictionary",
    }

    # TODO remove 'dim_threshold', 'max_dims' and 'typed_dict' in v1.7.0
    def __init__(
        self,
        n_parameter_samples=250,
        max_ensemble_size=50,
        max_win_len_prop=1,
        min_window=10,
        randomly_selected_params=50,
        bigrams=None,
        dim_threshold="deprecated",
        max_dims="deprecated",
        time_limit_in_minutes=0.0,
        contract_max_n_parameter_samples=np.inf,
        typed_dict="deprecated",
        train_estimate_method="loocv",
        n_jobs=1,
        random_state=None,
        max_channels=20,
        channel_threshold=0.85,
    ):
        self.n_parameter_samples = n_parameter_samples
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.min_window = min_window
        self.randomly_selected_params = randomly_selected_params
        self.bigrams = bigrams

        # multivariate
        self.dim_threshold = dim_threshold
        self.channel_threshold = channel_threshold
        if dim_threshold != "deprecated":
            warnings.warn(
                "The 'dim_threshold' parameter is deprecated and will be removed "
                "in v1.7.0. Use 'channel_threshold' instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.channel_threshold = dim_threshold
        self.max_dims = max_dims
        self.max_channels = max_channels
        if max_dims != "deprecated":
            warnings.warn(
                "The 'max_dims' parameter is deprecated and will be removed "
                "in v1.7.0. Use 'max_channels' instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.max_channels = max_dims

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_parameter_samples = contract_max_n_parameter_samples
        self.typed_dict = typed_dict
        if typed_dict != "deprecated":
            warnings.warn(
                "The 'typed_dict' parameter has no effect and will be removed "
                "in v1.7.0. Word counts are now stored as sorted arrays.",
                FutureWarning,
                stacklevel=2,
            )
        self.train_estimate_method = train_estimate_method
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0
        self.n_estimators_ = 0
        self.estimators_ = []
        self.weights_ = []

        self._word_lengths = [16, 14, 12, 10, 8]
        self._norm_options = [True, False]
        self._levels = [1, 2, 3]
        self._igb_options = [True, False]
        self._weight_sum = 0
        self._prev_parameters_x = []
        self._prev_parameters_y = []
        self._min_window = min_window
        super().__init__()

    def _fit(self, X, y, keep_train_preds=False):
        """Fit an ensemble on cases (X,y), where y is the target variable.

        Build an ensemble of base TDE classifiers from the training set (X,
        y), through an optimised selection over the parameter space to make a
        fixed size ensemble of the best.

        Parameters
        ----------
        X : 3D np.ndarray
            The training data shape = (n_cases, n_channels, n_timepoints).
        y : 1D np.ndarray
            The class labels shape = (n_cases).

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        if self.n_parameter_samples <= self.randomly_selected_params:
            warnings.warn(
                "TemporalDictionaryEnsemble warning: n_parameter_samples <= "
                "randomly_selected_params, ensemble member parameters will be fully "
                "randomly selected.",
                stacklevel=2,
            )

        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.estimators_ = []
        self.weights_ = []
        self._prev_parameters_x = []
        self._prev_parameters_y = []

        # Window length parameter space dependent on series length
        max_window_searches = self.n_timepoints_ / 4
        max_window = int(self.n_timepoints_ * self.max_win_len_prop)

        if self.min_window > max_window:
            self._min_window = max_window
            warnings.warn(
                f"TemporalDictionaryEnsemble warning: min_window = "
                f"{self.min_window} is larger than max_window = {max_window}."
                f" min_window has been set to {max_window}.",
                stacklevel=2,
            )

        win_inc = int((max_window - self._min_window) / max_window_searches)
        if win_inc < 1:
            win_inc = 1

        possible_parameters = self._unique_parameters(max_window, win_inc)
        # float array mirror of possible_parameters for the kernel ridge
        # parameter selection, kept in sync as parameters are popped
        candidate_parameters = np.array(possible_parameters, dtype=np.float64)
        num_classifiers = 0
        subsample_size = int(self.n_cases_ * 0.7)
        lowest_acc = 1
        lowest_acc_idx = 0

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0
        if time_limit > 0:
            n_parameter_samples = 0
            contract_max_n_parameter_samples = self.contract_max_n_parameter_samples
        else:
            n_parameter_samples = self.n_parameter_samples
            contract_max_n_parameter_samples = np.inf

        rng = check_random_state(self.random_state)

        if self.bigrams is None:
            if self.n_channels_ > 1:
                use_bigrams = False
            else:
                use_bigrams = True
        else:
            use_bigrams = self.bigrams

        # use time limit or n_parameter_samples if limit is 0
        while (
            (
                train_time < time_limit
                and num_classifiers < contract_max_n_parameter_samples
            )
            or num_classifiers < n_parameter_samples
        ) and len(possible_parameters) > 0:
            if num_classifiers < self.randomly_selected_params:
                idx = rng.randint(0, len(possible_parameters))
            else:
                # kernel ridge regression on standardised parameters, the
                # same computation as StandardScaler + KernelRidge(
                # kernel="poly", degree=1) but without the sklearn
                # per-call validation overhead
                preds = _kernel_ridge_preds(
                    np.array(self._prev_parameters_x, dtype=np.float64),
                    np.array(self._prev_parameters_y, dtype=np.float64),
                    candidate_parameters,
                )
                idx = rng.choice(np.flatnonzero(preds == preds.max()))

            parameters = possible_parameters.pop(idx)
            candidate_parameters = np.delete(candidate_parameters, idx, axis=0)

            while True:
                subsample = rng.choice(
                    self.n_cases_, size=subsample_size, replace=False
                )
                X_subsample = X[subsample]
                y_subsample = y[subsample]
                if len(np.unique(y_subsample)) > 1:
                    break

            # members are kept single threaded: the ensemble parallelises
            # over members in predict, so member-level threads would only
            # oversubscribe
            tde = IndividualTDE(
                *parameters,
                bigrams=use_bigrams,
                channel_threshold=self.channel_threshold,
                max_channels=self.max_channels,
                random_state=self.random_state,
            )
            # X_subsample/y_subsample are a subsample of the X/y this
            # ensemble already validated in full, so _fit is called
            # directly rather than through the public fit
            tde._fit(X_subsample, y_subsample)
            tde._subsample = subsample

            tde._accuracy = self._individual_train_acc(
                tde,
                y_subsample,
                subsample_size,
                0 if num_classifiers < self.max_ensemble_size else lowest_acc,
                keep_train_preds,
            )
            if tde._accuracy > 0:
                weight = math.pow(tde._accuracy, 4)
            else:
                weight = 0.000000001

            if num_classifiers < self.max_ensemble_size:
                if tde._accuracy < lowest_acc:
                    lowest_acc = tde._accuracy
                    lowest_acc_idx = num_classifiers
                self.weights_.append(weight)
                self.estimators_.append(tde)
            elif tde._accuracy > lowest_acc:
                self.weights_[lowest_acc_idx] = weight
                self.estimators_[lowest_acc_idx] = tde
                lowest_acc, lowest_acc_idx = self._worst_ensemble_acc()

            self._prev_parameters_x.append(parameters)
            self._prev_parameters_y.append(tde._accuracy)

            num_classifiers += 1
            train_time = time.time() - start_time

        self.n_estimators_ = len(self.estimators_)
        self._weight_sum = np.sum(self.weights_)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray
            The data to make predictions for, shape = (n_cases, n_channels,
            n_timepoints).

        Returns
        -------
        1D np.ndarray
            The predicted class labels shape = (n_cases).
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray
            The data to make predictions for, shape = (n_cases, n_channels,
            n_timepoints).

        Returns
        -------
        1D np.ndarray
            Predicted probabilities using the ordering in classes_, shape = (
            n_cases, n_classes_).

        """
        sums = np.zeros((X.shape[0], self.n_classes_))

        # each member's predict is dominated by nogil numba kernels, so
        # thread-based parallelism over members scales. X is validated once
        # by the public predict_proba wrapper, so members' _predict is
        # called directly. Results are gathered in member order, so the
        # aggregation below is identical for any n_jobs.
        if self._n_jobs > 1:
            all_preds = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(clf._predict)(X) for clf in self.estimators_
            )
        else:
            all_preds = [clf._predict(X) for clf in self.estimators_]

        for n, preds in enumerate(all_preds):
            for i in range(0, X.shape[0]):
                sums[i, self._class_dictionary[preds[i]]] += self.weights_[n]

        return sums / (np.ones(self.n_classes_) * self._weight_sum)

    def _fit_predict(self, X, y) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._fit_predict_proba(X, y)
            ]
        )

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        self._fit(X, y, keep_train_preds=True)

        results = np.zeros((self.n_cases_, self.n_classes_))
        divisors = np.zeros(self.n_cases_)

        if self.train_estimate_method.lower() == "loocv":
            for i, clf in enumerate(self.estimators_):
                subsample = clf._subsample
                preds = clf._train_predictions

                for n, pred in enumerate(preds):
                    results[subsample[n]][
                        self._class_dictionary[pred]
                    ] += self.weights_[i]
                    divisors[subsample[n]] += self.weights_[i]
        elif self.train_estimate_method.lower() == "oob":
            indices = range(self.n_cases_)
            for i, clf in enumerate(self.estimators_):
                oob = [n for n in indices if n not in clf._subsample]

                if len(oob) == 0:
                    continue

                # X[oob] is a subset of the already-validated X passed to
                # this method, so _predict is called directly as in
                # _predict_proba above
                preds = clf._predict(X[oob])

                for n, pred in enumerate(preds):
                    results[oob[n]][self._class_dictionary[pred]] += self.weights_[i]
                    divisors[oob[n]] += self.weights_[i]
        else:
            raise ValueError(
                "Invalid train_estimate_method. Available options: loocv, oob"
            )

        for i in range(self.n_cases_):
            results[i] = (
                np.ones(self.n_classes_) * (1 / self.n_classes_)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes_) * divisors[i])
            )

        return results

    def _worst_ensemble_acc(self):
        min_acc = 1.0
        min_acc_idx = 0

        for c, classifier in enumerate(self.estimators_):
            if classifier._accuracy < min_acc:
                min_acc = classifier._accuracy
                min_acc_idx = c

        return min_acc, min_acc_idx

    def _unique_parameters(self, max_window, win_inc):
        possible_parameters = [
            [win_size, word_len, normalise, levels, igb]
            for normalise in self._norm_options
            for win_size in range(self._min_window, max_window + 1, win_inc)
            for word_len in self._word_lengths
            for levels in self._levels
            for igb in self._igb_options
        ]

        return possible_parameters

    def _individual_train_acc(self, tde, y, train_size, lowest_acc, keep_train_preds):
        correct = 0
        required_correct = int(lowest_acc * train_size)

        # run the whole LOOCV in one numba call, computing each symmetric
        # pair intersection only once. The n x n similarity matrix is small
        # for typical subsample sizes; fall back to a per-case search for
        # very large n.
        if train_size <= _SYMMETRIC_LOOCV_MAX_N:
            _, y_codes = np.unique(y, return_inverse=True)
            n_done, correct, preds = loocv_train_acc(
                *tde._transformed_data, y_codes.astype(np.int64), required_correct
            )
            if keep_train_preds:
                for i in range(n_done):
                    tde._train_predictions.append(tde._class_vals[preds[i]])
            return -1 if correct == -1 else correct / train_size

        for i in range(train_size):
            if correct + train_size - i < required_correct:
                return -1

            c = tde._train_predict(i)

            if c == y[i]:
                correct += 1

            if keep_train_preds:
                tde._train_predictions.append(c)

        return correct / train_size

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            TemporalDictionaryEnsemble provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contracting
                    functionality
                "train_estimate" - used in some classifiers that set the
                    "capability:train_estimate" tag to True to allow for more efficient
                    testing when relevant parameters are available

        Returns
        -------
        dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        if parameter_set == "results_comparison":
            return {
                "n_parameter_samples": 10,
                "max_ensemble_size": 5,
                "randomly_selected_params": 5,
            }
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "contract_max_n_parameter_samples": 5,
                "max_ensemble_size": 2,
                "randomly_selected_params": 3,
            }
        else:
            return {
                "n_parameter_samples": 5,
                "max_ensemble_size": 2,
                "randomly_selected_params": 3,
            }


class IndividualTDE(BaseClassifier):
    """
    Single TDE classifier, an extension of the Bag of SFA Symbols (BOSS) model.

    Base classifier for the TDE classifier. Implementation of single TDE base model
    from [1]_.

    Overview: input "n" series of length "m" and IndividualTDE performs a SFA
    transform to form a sparse histogram of discretised words. The resulting
    histogram is used with the histogram intersection similarity function in a
    1-nearest neighbour.

    fit involves finding "n" histograms.

    predict uses 1 nearest neighbour with the histogram intersection similarity
    function.

    Parameters
    ----------
    window_size : int, default=10
        Size of the window to use in the SFA transform.
    word_length : int, default=8
        Length of word to use in the SFA transform.
    norm : bool, default=False
        Whether to normalize SFA words by dropping the first Fourier coefficient.
    levels : int, default=1
        The number of spatial pyramid levels for the SFA transform.
    igb : bool, default=False
        Whether to use Information Gain Binning (IGB) or
        Multiple Coefficient Binning (MCB) for the SFA transform.
    alphabet_size : int, default="deprecated"
        Has no effect: the alphabet size is fixed to 4.

        Deprecated and will be removed in v1.7.0.
    bigrams : bool, default=False
        Whether to record word bigrams in the SFA transform.
    channel_threshold : float, default=0.85
        Accuracy threshold as a proportion of the highest accuracy channel for words
        extracted from each channel. Only applicable for multivariate data.
    dim_threshold : float, default="deprecated"
        Deprecated alias for ``channel_threshold``. Will be removed in v1.7.0.
    max_channels : int, default=20
        Maximum number of channels words are extracted from. Only applicable for
        multivariate data.
    max_dims : int, default="deprecated"
        Deprecated alias for ``max_channels``. Will be removed in v1.7.0.
    typed_dict : bool, default="deprecated"
        Has no effect: word counts are now stored as sorted arrays.

        Deprecated and will be removed in v1.7.0.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `predict`. `fit` is
        single threaded. ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for the random number generator.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of channels per case.
    n_timepoints_ : int
        The length of each series.

    See Also
    --------
    TemporalDictionaryEnsemble, SFA
        TDE extends BOSS and uses SFA.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/IndividualTDE.java>`_.

    References
    ----------
    .. [1] Matthew Middlehurst, James Large, Gavin Cawley and Anthony Bagnall
        "The Temporal Dictionary Ensemble (TDE) Classifier for Time Series
        Classification", in proceedings of the European Conference on Machine Learning
        and Principles and Practice of Knowledge Discovery in Databases, 2020.

    Examples
    --------
    >>> from aeon.classification.dictionary_based import IndividualTDE
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = IndividualTDE()
    >>> clf.fit(X_train, y_train)
    IndividualTDE(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    # TODO remove 'dim_threshold', 'alphabet_size', 'max_dims' and 'typed_dict'
    # in v1.7.0
    def __init__(
        self,
        window_size=10,
        word_length=8,
        norm=False,
        levels=1,
        igb=False,
        alphabet_size="deprecated",
        bigrams=True,
        dim_threshold="deprecated",
        max_dims="deprecated",
        typed_dict="deprecated",
        n_jobs=1,
        random_state=None,
        max_channels=20,
        channel_threshold=0.85,
    ):
        self.window_size = window_size
        self.word_length = word_length
        self.norm = norm
        self.levels = levels
        self.igb = igb
        self.alphabet_size = alphabet_size
        if alphabet_size != "deprecated":
            warnings.warn(
                "The 'alphabet_size' parameter has no effect and will be "
                "removed in v1.7.0. The alphabet size is fixed to 4.",
                FutureWarning,
                stacklevel=2,
            )
        self.bigrams = bigrams

        # multivariate
        self.dim_threshold = dim_threshold
        self.channel_threshold = channel_threshold
        if dim_threshold != "deprecated":
            warnings.warn(
                "The 'dim_threshold' parameter is deprecated and will be removed "
                "in v1.7.0. Use 'channel_threshold' instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.channel_threshold = dim_threshold
        self.max_dims = max_dims
        self.max_channels = max_channels
        if max_dims != "deprecated":
            warnings.warn(
                "The 'max_dims' parameter is deprecated and will be removed "
                "in v1.7.0. Use 'max_channels' instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.max_channels = max_dims

        self.typed_dict = typed_dict
        if typed_dict != "deprecated":
            warnings.warn(
                "The 'typed_dict' parameter has no effect and will be removed "
                "in v1.7.0. Word counts are now stored as sorted arrays.",
                FutureWarning,
                stacklevel=2,
            )
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0

        self._transformers = []
        self._transformed_data = []
        self._class_vals = []
        self._channels = []
        self._highest_channel_bit = 0
        self._accuracy = 0
        self._subsample = []
        self._train_predictions = []

        super().__init__()

    def _fit(self, X, y):
        """Fit a single base TDE classifier on n_cases cases (X,y).

        Parameters
        ----------
        X : 3D np.ndarray
            The training data shape = (n_cases, n_channels, n_timepoints).
        y : 1D np.ndarray
            The training labels, shape = (n_cases).

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)
        self._class_vals = y

        # select channels using accuracy estimate if multivariate
        if self.n_channels_ > 1:
            self._channels, self._transformers = self._select_channels(X, y)
            channel_words = [
                self._transformers[i].transform(self._transformers[i]._fit_X)
                for i in range(len(self._channels))
            ]
            self._transformed_data = self._combine_channel_bags(
                channel_words, self._channels, self.n_cases_
            )
        else:
            self._transformers.append(
                _TDE_SFA(
                    word_length=self.word_length,
                    window_size=self.window_size,
                    norm=self.norm,
                    levels=self.levels,
                    binning_method="information-gain" if self.igb else "equi-depth",
                    bigrams=self.bigrams,
                )
            )
            self._transformed_data = self._transformers[0].fit_transform(
                np.ascontiguousarray(X[:, 0, :]), y
            )

        self._clear_transformer_fit_cache()

    def _predict(self, X):
        """Predict class values of all instances in X.

        Parameters
        ----------
        X : 3D np.ndarray
            The data to make predictions for, shape = (n_cases, n_channels,
            n_timepoints).

        Returns
        -------
        1D np.ndarray
            The predicted class labels shape = (n_cases).
        """
        n_cases = X.shape[0]

        if self.n_channels_ > 1:
            channel_words = [
                self._transformers[i].transform(np.ascontiguousarray(X[:, channel, :]))
                for i, channel in enumerate(self._channels)
            ]
            test_bags = self._combine_channel_bags(
                channel_words, self._channels, n_cases
            )
        else:
            test_bags = self._transformers[0].transform(
                np.ascontiguousarray(X[:, 0, :])
            )

        # all test-vs-train similarities in numba calls that release the GIL,
        # then a cheap per-case tie-break loop. With n_jobs > 1 the test
        # cases are chunked across threads; chunk results are stacked in
        # order, so the similarities are identical for any n_jobs.
        keys1, keys2, counts, t_offsets = test_bags
        if self._n_jobs > 1 and n_cases > 1:
            chunks = np.array_split(np.arange(n_cases), min(self._n_jobs, n_cases))
            sims = np.vstack(
                Parallel(n_jobs=self._n_jobs, prefer="threads")(
                    delayed(nn_similarities_all)(
                        *self._transformed_data,
                        keys1,
                        keys2,
                        counts,
                        t_offsets[chunk[0] : chunk[-1] + 2],
                    )
                    for chunk in chunks
                )
            )
        else:
            sims = nn_similarities_all(
                *self._transformed_data, keys1, keys2, counts, t_offsets
            )

        if isinstance(self.random_state, (int, np.integer)) and not isinstance(
            self.random_state, bool
        ):
            # with an integer seed every case's tie-break generator yields
            # the same sequence, so one precomputed draw pool resolves all
            # cases inside numba, exactly as per-case generators would
            draws = check_random_state(self.random_state).random(sims.shape[1])
            nn_idx = nn_tie_break(sims, draws)
            classes = [self._class_vals[nn_idx[i]] for i in range(n_cases)]
        else:
            # unseeded or shared generators consume draws across cases, so
            # tie events must be resolved sequentially; the running maximum
            # is draw-independent, so tie-free cases are resolved in numba
            nn0, has_tie = nn_first_max(sims)
            classes = [
                self._nn_from_sims(sims[i]) if has_tie[i] else self._class_vals[nn0[i]]
                for i in range(n_cases)
            ]
        return np.array(classes)

    def _clear_transformer_fit_cache(self):
        for transformer in self._transformers:
            if hasattr(transformer, "_fit_X"):
                transformer._fit_X = None
            if hasattr(transformer, "_fit_mft"):
                transformer._fit_mft = None

    def _nn_from_sims(self, sims):
        # the rng is only consumed on similarity ties, so construct it
        # lazily: seeding a RandomState per test case is far more expensive
        # than the tie-break draws themselves
        rng = None
        best_sim = -1
        nn = None
        for n in range(len(sims)):
            sim = sims[n]
            if sim > best_sim:
                best_sim = sim
                nn = self._class_vals[n]
            elif sim == best_sim:
                if rng is None:
                    rng = check_random_state(self.random_state)
                if rng.random() < 0.5:
                    nn = self._class_vals[n]

        return nn

    def _combine_channel_bags(self, channel_bags, channels, n_cases):
        # per-channel bags are already sorted, so a numba k-way merge
        # builds the combined sorted bags without any re-sorting
        all_k1 = np.concatenate([b[0] for b in channel_bags])
        all_k2 = np.concatenate([b[1] for b in channel_bags])
        all_v = np.concatenate([b[2] for b in channel_bags])
        channel_case_offsets = np.vstack([b[3] for b in channel_bags])
        sizes = np.array([len(b[0]) for b in channel_bags], dtype=np.int64)
        channel_starts = np.zeros(len(channel_bags), dtype=np.int64)
        channel_starts[1:] = np.cumsum(sizes)[:-1]

        return combine_channel_bags(
            all_k1,
            all_k2,
            all_v,
            channel_case_offsets,
            channel_starts,
            np.asarray(channels, dtype=np.int64),
            self.levels,
            self._highest_channel_bit,
        )

    def _select_channels(self, X, y):
        self._highest_channel_bit = (math.ceil(math.log2(self.n_channels_))) + 1
        channel_accs = []
        transformers = []

        _, y_codes = np.unique(y, return_inverse=True)
        y_codes = y_codes.astype(np.int64)

        # select channels based on reduced bag size accuracy
        for channel in range(self.n_channels_):
            self._channels.append(channel)
            transformers.append(
                _TDE_SFA(
                    word_length=self.word_length,
                    window_size=self.window_size,
                    norm=self.norm,
                    levels=self.levels,
                    binning_method="information-gain" if self.igb else "equi-depth",
                    bigrams=self.bigrams,
                    keep_binning_dft=True,
                )
            )

            X_channel = np.ascontiguousarray(X[:, channel, :])

            transformers[channel].fit(X_channel, y)
            sfa = transformers[channel].binning_bags()
            transformers[channel].keep_binning_dft = False
            transformers[channel]._binning_dft = None

            if self.n_cases_ <= _SYMMETRIC_LOOCV_MAX_N:
                # whole LOOCV in one numba call, each symmetric pair
                # intersection computed once
                _, correct, _ = loocv_train_acc(*sfa, y_codes, 0)
            else:
                correct = 0
                for n in range(self.n_cases_):
                    if self._train_predict(n, sfa) == y[n]:
                        correct = correct + 1

            channel_accs.append(correct)

        max_acc = max(channel_accs)

        channels = []
        fin_transformers = []
        for channel in range(self.n_channels_):
            if channel_accs[channel] >= max_acc * self.channel_threshold:
                channels.append(channel)
                fin_transformers.append(transformers[channel])

        if len(channels) > self.max_channels:
            rng = check_random_state(self.random_state)
            idx = rng.choice(len(channels), self.max_channels, replace=False).tolist()
            channels = [channels[i] for i in idx]
            fin_transformers = [fin_transformers[i] for i in idx]

        return channels, fin_transformers

    def _train_predict(self, train_num, bags=None):
        if bags is None:
            bags = self._transformed_data

        nn_idx = nn_predict_loocv(*bags, train_num)
        return self._class_vals[nn_idx] if nn_idx >= 0 else None


def histogram_intersection(first, second):
    """Find the similarity between two histograms using the histogram intersection.

    This similarity function is designed for sparse histograms represented as
    a dictionary or numba Dict, but can accept arrays in dense format.

    Parameters
    ----------
    first : dict, numba.Dict or 1D array of integers
        First histogram used in the similarity measurement.
    second : dict, numba.Dict or 1D array of integers
        Second histogram that will be used to measure similarity to `first`.

    Returns
    -------
    sim : int
        The histogram intersection similarity (the sum of minimum counts over
        shared words) between the first and second histograms.
    """
    if isinstance(first, dict):
        sim = 0
        for word, val_a in first.items():
            val_b = second.get(word, 0)
            sim += min(val_a, val_b)
        return sim
    elif isinstance(first, Dict):
        return _histogram_intersection_dict(first, second)
    else:
        return np.sum(
            [
                0 if first[n] == 0 else np.minimum(first[n], second[n])
                for n in range(len(first))
            ]
        )


@njit(fastmath=True, cache=True)
def _histogram_intersection_dict(first, second):
    sim = 0
    for word, val_a in first.items():
        val_b = second.get(word, types.uint32(0))
        sim += min(val_a, val_b)
    return sim


_DBL_MAX = sys.float_info.max

ALPHABET_SIZE = 4
LETTER_BITS = 2


class _TDE_SFA:
    """Symbolic Fourier Approximation transform, TDE feature set only.

    Parameters
    ----------
    word_length : int, default=8
        Number of Fourier coefficients (letters) per word.
    window_size : int, default=12
        Length of the sliding window.
    norm : bool, default=False
        If True, drop the first Fourier coefficient pair (mean normalise).
    levels : int, default=1
        Number of spatial pyramid levels.
    binning_method : str, default="equi-depth"
        "equi-depth" (MCB) or "information-gain" (IGB).
    bigrams : bool, default=False
        Whether to add bigram words (pairs of words window_size apart).
    keep_binning_dft : bool, default=False
        Keep the binning DFT after fit so `binning_bags()` can build the
        reduced bags used by TDE's multivariate channel selection.

    Attributes
    ----------
    breakpoints : 2D np.ndarray (word_length, 4)
        Discretisation boundaries per letter, last column is DBL_MAX.
    """

    def __init__(
        self,
        word_length=8,
        window_size=12,
        norm=False,
        levels=1,
        binning_method="equi-depth",
        bigrams=False,
        keep_binning_dft=False,
    ):
        if word_length < 1:
            raise ValueError("word_length must be at least 1")
        if not 1 <= levels <= 3:
            raise ValueError("levels must be 1, 2 or 3 (the TDE parameter space)")
        if binning_method not in ("equi-depth", "information-gain"):
            raise ValueError(
                "binning_method must be 'equi-depth' or 'information-gain'"
            )

        self.word_length = word_length
        self.window_size = window_size
        self.norm = norm
        self.levels = levels
        self.binning_method = binning_method
        self.bigrams = bigrams
        self.keep_binning_dft = keep_binning_dft

        self.word_bits = word_length * LETTER_BITS
        if self.word_bits * (2 if bigrams else 1) > 64:
            raise ValueError("words (and bigrams) must fit in 64 bits")

        # number of Fourier values kept per window (even, pairs of real/imag)
        self.dft_length = word_length + word_length % 2
        # extra leading pair computed then dropped when norm is used
        self.norm_offset = 2 if norm else 0
        self.inverse_sqrt_win_size = 1.0 / math.sqrt(window_size)

        self.breakpoints = None
        self.n_timepoints = 0
        self._fit_X = None
        self._fit_mft = None
        self._binning_dft = None

    def fit(self, X, y=None):
        """Learn breakpoints from X (2D or squeezable 3D array)."""
        X = self._check_X(X)
        n_cases, self.n_timepoints = X.shape
        if self.window_size > self.n_timepoints:
            raise ValueError("window_size larger than series length")
        if self.binning_method == "information-gain" and y is None:
            raise ValueError("y is required for information gain binning")

        mft = _mft_all(
            X,
            self.window_size,
            self.dft_length + self.norm_offset,
            self.norm_offset,
            self.inverse_sqrt_win_size,
        )

        num_windows_per_inst = int(math.ceil(self.n_timepoints / self.window_size))
        idx = np.empty(num_windows_per_inst, dtype=np.int64)
        for i in range(num_windows_per_inst - 1):
            idx[i] = i * self.window_size
        idx[-1] = self.n_timepoints - self.window_size

        direct_binning_dft = None
        if self.binning_method == "information-gain" or self.keep_binning_dft:
            direct_binning_dft = _binning_dft_all(
                X,
                self.window_size,
                self.dft_length,
                self.norm,
                self.inverse_sqrt_win_size,
                num_windows_per_inst,
            )

        binning_dft = (
            direct_binning_dft
            if self.binning_method == "information-gain"
            else mft[:, idx, :]
        )

        flat = np.ascontiguousarray(
            binning_dft.reshape(n_cases * num_windows_per_inst, mft.shape[2])
        )
        if self.binning_method == "information-gain":
            # one label per binning window
            self.breakpoints = self._igb(flat, np.repeat(y, num_windows_per_inst))
        else:
            self.breakpoints = self._mcb_equi_depth(flat)

        self._fit_X = X
        self._fit_mft = mft
        self._binning_dft = (
            np.ascontiguousarray(direct_binning_dft) if self.keep_binning_dft else None
        )
        return self

    def transform(self, X):
        """Transform X into bags of words.

        Returns
        -------
        (keys1, keys2, counts, offsets) :
            Concatenated per-case bags. Case i's bag is rows
            offsets[i]:offsets[i+1], sorted lexicographically by
            (keys1, keys2). counts is uint32.
        """
        X = self._check_X(X)
        if self._fit_X is not None and X is self._fit_X:
            mft = self._fit_mft
        else:
            mft = _mft_all(
                X,
                self.window_size,
                self.dft_length + self.norm_offset,
                self.norm_offset,
                self.inverse_sqrt_win_size,
            )
        return self._bags(mft)

    def fit_transform(self, X, y=None):
        """Fit and transform sharing the MFT computation."""
        self.fit(X, y)
        return self._bags(self._fit_mft)

    def binning_bags(self):
        """Bags built from the binning DFT (reduced bags for channel selection)."""
        if self._binning_dft is None:
            raise ValueError("fit with keep_binning_dft=True first")
        return self._bags(self._binning_dft)

    def _bags(self, dfts):
        return _bags_from_dft(
            dfts,
            self.breakpoints,
            self.word_length,
            self.word_bits,
            self.window_size,
            self.n_timepoints,
            self.levels,
            self.bigrams,
        )

    def _check_X(self, X):
        if X.ndim == 3:
            if X.shape[1] != 1:
                raise ValueError("only univariate input, slice channels in TDE")
            X = X.reshape(X.shape[0], X.shape[2])
        return np.ascontiguousarray(X, dtype=np.float64)

    def _mcb_equi_depth(self, dft):
        total = dft.shape[0]
        breakpoints = np.zeros((self.word_length, ALPHABET_SIZE))
        target_bin_depth = total / ALPHABET_SIZE

        for letter in range(self.word_length):
            # 2dp rounding retained from the original implementation
            column = np.sort(np.rint(dft[:, letter] * 100) / 100)
            bin_index = 0.0
            for bp in range(ALPHABET_SIZE - 1):
                bin_index += target_bin_depth
                breakpoints[letter, bp] = column[int(bin_index)]

        breakpoints[:, ALPHABET_SIZE - 1] = _DBL_MAX
        return breakpoints

    def _igb(self, dft, y):
        y = np.asarray(y)
        _, y_codes = np.unique(y, return_inverse=True)
        dft = dft[:, : self.word_length].astype(np.float32).astype(np.float64)
        thresholds, n_thresholds = _igb_all(
            dft,
            y_codes.astype(np.int64),
            int(y_codes.max()) + 1,
        )

        breakpoints = np.full((self.word_length, ALPHABET_SIZE), _DBL_MAX)
        for letter in range(self.word_length):
            for bp in range(n_thresholds[letter]):
                breakpoints[letter, bp] = thresholds[letter, bp]
        return np.sort(breakpoints, axis=1)


@njit(fastmath=True, cache=True, nogil=True)
def _incremental_stds(series, end, window_size):
    stds = np.zeros(end)
    series_sum = 0.0
    square_sum = 0.0
    for i in range(window_size):
        series_sum += series[i]
        square_sum += series[i] * series[i]

    r_window_length = 1.0 / window_size
    mean = series_sum * r_window_length
    buf = math.sqrt(square_sum * r_window_length - mean * mean)
    stds[0] = buf if buf > 1e-8 else 1.0

    for w in range(1, end):
        series_sum += series[w + window_size - 1] - series[w - 1]
        mean = series_sum * r_window_length
        square_sum += (
            series[w + window_size - 1] * series[w + window_size - 1]
            - series[w - 1] * series[w - 1]
        )
        buf = math.sqrt(square_sum * r_window_length - mean * mean)
        stds[w] = buf if buf > 1e-8 else 1.0

    return stds


@njit(fastmath=True, cache=True, nogil=True)
def _mft_all(X, window_size, length, norm_offset, inverse_sqrt_win_size):
    """Normalised sliding-window Fourier coefficients for every case.

    Returns a (n_cases, n_windows, length - norm_offset) array: the first
    norm_offset values (the first coefficient pair when norm is used) are
    dropped from the output.
    """
    n_cases, n_timepoints = X.shape
    end = max(1, n_timepoints - window_size + 1)
    half = length // 2

    phis = np.zeros(length)
    for i in range(half):
        phis[i * 2] = math.cos(2 * math.pi * (-i) / window_size)
        phis[i * 2 + 1] = -math.sin(2 * math.pi * (-i) / window_size)

    # cos/sin(2*pi*t*i/w) is periodic in (t*i) mod w, so one table of size
    # window_size replaces all trig calls in the first-window DFT
    cos_t = np.empty(window_size)
    sin_t = np.empty(window_size)
    for k in range(window_size):
        angle = 2 * math.pi * k / window_size
        cos_t[k] = math.cos(angle)
        sin_t[k] = math.sin(angle)

    out_len = length - norm_offset
    out = np.zeros((n_cases, end, out_len))
    mft = np.zeros(length)

    for c in range(n_cases):
        series = X[c]
        stds = _incremental_stds(series, end, window_size)

        # first window: direct DFT, O(window_size * length)
        for i in range(half):
            step = i % window_size
            real = 0.0
            imag = 0.0
            idx = 0
            for t in range(window_size):
                real += series[t] * cos_t[idx]
                imag += -series[t] * sin_t[idx]
                idx += step
                if idx >= window_size:
                    idx -= window_size
            mft[i * 2] = real
            mft[i * 2 + 1] = imag

        factor = inverse_sqrt_win_size / stds[0]
        for j in range(out_len):
            out[c, 0, j] = mft[norm_offset + j] * factor

        # remaining windows: incremental MFT update
        for w in range(1, end):
            diff = series[w + window_size - 1] - series[w - 1]
            for i2 in range(0, length, 2):
                real = mft[i2] + diff
                imag = mft[i2 + 1]
                mft[i2] = real * phis[i2] - imag * phis[i2 + 1]
                mft[i2 + 1] = real * phis[i2 + 1] + phis[i2] * imag

            factor = inverse_sqrt_win_size / stds[w]
            for j in range(out_len):
                out[c, w, j] = mft[norm_offset + j] * factor

    return out


@njit(fastmath=True, cache=True, nogil=True)
def _binning_dft_all(
    X, window_size, dft_length, norm, inverse_sqrt_win_size, num_windows_per_inst
):
    n_cases, n_timepoints = X.shape
    start = 2 if norm else 0
    output_length = start + dft_length
    c = start // 2

    # cos/sin(2*pi*n*i/w) is periodic in (n*i) mod w, so one table of size
    # window_size replaces all trig calls
    cos_t = np.empty(window_size)
    sin_t = np.empty(window_size)
    for k in range(window_size):
        angle = 2 * math.pi * k / window_size
        cos_t[k] = math.cos(angle)
        sin_t[k] = math.sin(angle)

    out = np.zeros((n_cases, num_windows_per_inst, dft_length))
    for case in range(n_cases):
        series = X[case]
        for window in range(num_windows_per_inst):
            if window == num_windows_per_inst - 1:
                offset = n_timepoints - window_size
            else:
                offset = window * window_size

            series_sum = 0.0
            for n in range(window_size):
                series_sum += series[offset + n]

            mean = series_sum / window_size
            squared_deviation_sum = 0.0
            for n in range(window_size):
                deviation = series[offset + n] - mean
                squared_deviation_sum += deviation * deviation

            std = math.sqrt(squared_deviation_sum / window_size)
            if std == 0.0:
                std = 1.0
            factor = inverse_sqrt_win_size / std

            for i in range(c, output_length // 2):
                step = i % window_size
                real = 0.0
                imag = 0.0
                idx = 0
                for n in range(window_size):
                    value = series[offset + n]
                    real += value * cos_t[idx]
                    imag += -value * sin_t[idx]
                    idx += step
                    if idx >= window_size:
                        idx -= window_size
                out[case, window, (i - c) * 2] = real * factor
                out[case, window, (i - c) * 2 + 1] = imag * factor

    return out


@njit(cache=True, nogil=True)
def _bags_from_dft(
    dfts,
    breakpoints,
    word_length,
    word_bits,
    window_size,
    n_timepoints,
    levels,
    bigrams,
):
    """Words and aggregated bags for all cases from their window DFTs.

    Numerosity reduction is always applied; the alphabet is fixed at 4
    (2 bits per letter). Output bags are sorted lexicographically by
    (key1, key2).

    Bag events carry no explicit values: a bigram always counts 1 and a
    pyramid unigram counts 2**level, which is recoverable from its quadrant.
    With levels <= 3 a unigram event packs into a single int64
    ((word << 3) | quadrant), so per-case bags reduce to plain np.sort of
    key arrays followed by run-length aggregation.

    - levels == 1: unigram words and packed bigrams share one key space
      (key2 = 0), exactly like the flat typed-dict SFA, so a bigram with
      previous word 0 merges with the unigram of the same value.
    - levels > 1: unigrams are packed (word << 3) | quadrant; bigrams are
      kept separately with key2 = -1 and merged in during aggregation.
    """
    n_cases, n_windows, _ = dfts.shape
    events_per_case = n_windows * (levels + (1 if bigrams else 0))

    keys1 = np.empty(n_cases * events_per_case, dtype=np.int64)
    keys2 = np.empty(n_cases * events_per_case, dtype=np.int64)
    counts = np.empty(n_cases * events_per_case, dtype=np.uint32)
    offsets = np.zeros(n_cases + 1, dtype=np.int64)

    words = np.zeros(n_windows, dtype=np.int64)
    # when levels == 1 bigrams share the unigram key space and array
    uni = np.empty(n_windows * (levels + (1 if bigrams else 0)), dtype=np.int64)
    big = np.empty(n_windows if bigrams else 0, dtype=np.int64)

    pos = 0
    for c in range(n_cases):
        # one word per window; letter = number of breakpoints below the
        # value (branchless, alphabet 4, last breakpoint is DBL_MAX)
        for wi in range(n_windows):
            word = np.int64(0)
            for i in range(word_length):
                v = dfts[c, wi, i]
                letter = (
                    np.int64(v > breakpoints[i, 0])
                    + np.int64(v > breakpoints[i, 1])
                    + np.int64(v > breakpoints[i, 2])
                )
                word = (word << LETTER_BITS) | letter
            words[wi] = word

        # emit key events, numerosity reduction always on
        n_u = 0
        n_b = 0
        last_word = np.int64(-1)
        repeat_words = 0
        for wi in range(n_windows):
            word = words[wi]

            if word == last_word:
                repeat_words += 1
            else:
                if levels > 1:
                    window_ind = wi - repeat_words // 2
                    start = 0
                    for level in range(levels):
                        num_quadrants = 2**level
                        quadrant = start + (window_ind + window_size // 2) // (
                            n_timepoints // num_quadrants
                        )
                        uni[n_u] = (word << 3) | quadrant
                        n_u += 1
                        start += num_quadrants
                else:
                    uni[n_u] = word
                    n_u += 1
                last_word = word
                repeat_words = 0

            if bigrams and wi - window_size >= 0:
                bigram = (words[wi - window_size] << word_bits) | word
                if levels > 1:
                    big[n_b] = bigram
                    n_b += 1
                else:
                    # shared key space with unigrams, matching the flat
                    # typed-dict SFA
                    uni[n_u] = bigram
                    n_u += 1

        su = np.sort(uni[:n_u])
        if levels > 1:
            sb = np.sort(big[:n_b])
        else:
            sb = big[:0]
            n_b = 0

        # merge-aggregate the two sorted streams; for a shared key1 the
        # bigram (key2 = -1) sorts before any unigram (key2 >= 0)
        i = 0
        j = 0
        while i < n_u or j < n_b:
            if j < n_b and (i >= n_u or sb[j] <= su[i] >> 3):
                # bigram run
                bk = sb[j]
                run = 1
                j += 1
                while j < n_b and sb[j] == bk:
                    run += 1
                    j += 1
                keys1[pos] = bk
                keys2[pos] = -1
                counts[pos] = run
                pos += 1
            else:
                # unigram run
                uk = su[i]
                run = 1
                i += 1
                while i < n_u and su[i] == uk:
                    run += 1
                    i += 1
                if levels > 1:
                    quadrant = uk & 7
                    # weight 2**level of the quadrant: 0 -> 1, 1-2 -> 2,
                    # 3-6 -> 4
                    if quadrant == 0:
                        weight = 1
                    elif quadrant <= 2:
                        weight = 2
                    else:
                        weight = 4
                    keys1[pos] = uk >> 3
                    keys2[pos] = quadrant
                    counts[pos] = run * weight
                else:
                    keys1[pos] = uk
                    keys2[pos] = 0
                    counts[pos] = run
                pos += 1

        offsets[c + 1] = pos

    return keys1[:pos].copy(), keys2[:pos].copy(), counts[:pos].copy(), offsets


@njit(cache=True)
def _entropy(class_counts, n):
    h = 0.0
    for k in range(len(class_counts)):
        if class_counts[k] > 0:
            p = class_counts[k] / n
            h -= p * math.log(p)
    return h


@njit(cache=True)
def _best_split(xs, ys, start, end, n_classes, n_total):
    """Best entropy split of sorted segment [start, end).

    Returns (improvement, split_pos, threshold); split_pos is the index of
    the last element of the left child, or -1 if no valid split exists.
    Improvement is sklearn's weighted impurity decrease (n_node / n_total) *
    (H - weighted child H), so it is comparable across nodes.
    """
    n_node = end - start
    if n_node < 2:
        return -1.0, -1, 0.0

    counts = np.zeros(n_classes)
    for i in range(start, end):
        counts[ys[i]] += 1
    h_node = _entropy(counts, n_node)
    if h_node <= 1e-12:
        return -1.0, -1, 0.0

    left = np.zeros(n_classes)
    right = counts.copy()
    best_gain = -1.0
    best_pos = -1
    best_thr = 0.0
    n_left = 0

    for i in range(start, end - 1):
        left[ys[i]] += 1
        right[ys[i]] -= 1
        n_left += 1
        if xs[i + 1] > xs[i]:
            n_right = n_node - n_left
            weighted = (
                n_left * _entropy(left, n_left) + n_right * _entropy(right, n_right)
            ) / n_node
            gain = (n_node / n_total) * (h_node - weighted)
            if gain > best_gain:
                best_gain = gain
                best_pos = i
                thr = (xs[i] + xs[i + 1]) / 2.0
                # guard against midpoint rounding up to the right value
                if thr == xs[i + 1]:
                    thr = xs[i]
                best_thr = thr

    return best_gain, best_pos, best_thr


@njit(cache=True)
def _igb_all(dft, y_codes, n_classes):
    """Information gain binning for every letter, alphabet fixed at 4.

    Best-first growth of an entropy decision tree on one feature until 4
    leaves (or no further valid splits), depth-limited to 2, the same
    procedure sklearn's DecisionTreeClassifier uses with max_leaf_nodes=4
    and max_depth=2.
    """
    max_depth = 2
    n_letters = dft.shape[1]
    n_total = dft.shape[0]

    thresholds = np.zeros((n_letters, ALPHABET_SIZE - 1))
    n_thresholds = np.zeros(n_letters, dtype=np.int64)

    max_cand = 2 * ALPHABET_SIZE
    c_start = np.zeros(max_cand, dtype=np.int64)
    c_end = np.zeros(max_cand, dtype=np.int64)
    c_depth = np.zeros(max_cand, dtype=np.int64)
    c_pos = np.zeros(max_cand, dtype=np.int64)
    c_thr = np.zeros(max_cand)
    c_gain = np.zeros(max_cand)
    c_active = np.zeros(max_cand, dtype=np.bool_)

    for letter in range(n_letters):
        col = dft[:, letter]
        order = np.argsort(col)
        xs = col[order]
        ys = y_codes[order]

        n_cand = 0
        c_active[:] = False

        gain, pos, thr = _best_split(xs, ys, 0, n_total, n_classes, n_total)
        if pos >= 0:
            c_start[0] = 0
            c_end[0] = n_total
            c_depth[0] = 0
            c_pos[0] = pos
            c_thr[0] = thr
            c_gain[0] = gain
            c_active[0] = True
            n_cand = 1

        n_leaves = 1
        n_thr = 0
        while n_leaves < ALPHABET_SIZE:
            # pick the active candidate with the highest improvement
            best = -1
            best_gain = -1.0
            for k in range(n_cand):
                if c_active[k] and c_gain[k] > best_gain:
                    best_gain = c_gain[k]
                    best = k
            if best < 0:
                break

            c_active[best] = False
            thresholds[letter, n_thr] = c_thr[best]
            n_thr += 1
            n_leaves += 1

            # evaluate the two children as new candidates
            child_depth = c_depth[best] + 1
            if child_depth < max_depth:
                s = c_start[best]
                e = c_end[best]
                mid = c_pos[best] + 1
                for lo, hi in ((s, mid), (mid, e)):
                    gain, pos, thr = _best_split(xs, ys, lo, hi, n_classes, n_total)
                    if pos >= 0 and n_cand < max_cand:
                        c_start[n_cand] = lo
                        c_end[n_cand] = hi
                        c_depth[n_cand] = child_depth
                        c_pos[n_cand] = pos
                        c_thr[n_cand] = thr
                        c_gain[n_cand] = gain
                        c_active[n_cand] = True
                        n_cand += 1

        n_thresholds[letter] = n_thr

    return thresholds, n_thresholds


@njit(cache=True, nogil=True)
def _histogram_intersection(keys1, keys2, counts, a0, a1, b0, b1):
    """Merge intersection of two sorted bag segments (sum of min counts)."""
    sim = 0
    i, j = a0, b0
    while i < a1 and j < b1:
        ka1, ka2 = keys1[i], keys2[i]
        kb1, kb2 = keys1[j], keys2[j]
        if ka1 == kb1 and ka2 == kb2:
            sim += min(counts[i], counts[j])
            i += 1
            j += 1
        elif ka1 < kb1 or (ka1 == kb1 and ka2 < kb2):
            i += 1
        else:
            j += 1
    return sim


@njit(cache=True, nogil=True)
def combine_channel_bags(
    all_k1,
    all_k2,
    all_v,
    channel_case_offsets,
    channel_starts,
    channels,
    levels,
    highest_channel_bit,
):
    """Merge per-channel bags into combined multivariate bags.

    Every per-channel bag is already sorted by (key1, key2) and the
    channel rewrite of key2 ((key2 << highest_channel_bit) | channel for levels
    > 1, channel otherwise) is monotone, so each stream stays sorted and a
    k-way merge produces the combined bag in lexicographic order. Keys
    from different channels can never be equal (the channel is in key2), so
    no aggregation is needed.

    all_* are the per-channel arrays concatenated in channel order,
    channel_starts[d] is where channel d's block begins and
    channel_case_offsets[d] is channel d's per-case offsets array.
    """
    n_channels = channel_case_offsets.shape[0]
    n_cases = channel_case_offsets.shape[1] - 1
    total = all_k1.shape[0]

    out_k1 = np.empty(total, dtype=np.int64)
    out_k2 = np.empty(total, dtype=np.int64)
    out_v = np.empty(total, dtype=np.uint32)
    offsets = np.zeros(n_cases + 1, dtype=np.int64)

    ptr = np.empty(n_channels, dtype=np.int64)
    end = np.empty(n_channels, dtype=np.int64)
    cur1 = np.empty(n_channels, dtype=np.int64)
    cur2 = np.empty(n_channels, dtype=np.int64)

    pos = 0
    for n in range(n_cases):
        active = 0
        for channel in range(n_channels):
            ptr[channel] = channel_starts[channel] + channel_case_offsets[channel, n]
            end[channel] = (
                channel_starts[channel] + channel_case_offsets[channel, n + 1]
            )
            if ptr[channel] < end[channel]:
                cur1[channel] = all_k1[ptr[channel]]
                if levels > 1:
                    cur2[channel] = (
                        all_k2[ptr[channel]] << highest_channel_bit
                    ) | channels[channel]
                else:
                    cur2[channel] = channels[channel]
                active += 1

        while active > 0:
            best = -1
            for channel in range(n_channels):
                if ptr[channel] < end[channel] and (
                    best < 0
                    or cur1[channel] < cur1[best]
                    or (cur1[channel] == cur1[best] and cur2[channel] < cur2[best])
                ):
                    best = channel

            out_k1[pos] = cur1[best]
            out_k2[pos] = cur2[best]
            out_v[pos] = all_v[ptr[best]]
            pos += 1

            ptr[best] += 1
            if ptr[best] < end[best]:
                cur1[best] = all_k1[ptr[best]]
                if levels > 1:
                    cur2[best] = (all_k2[ptr[best]] << highest_channel_bit) | channels[
                        best
                    ]
                else:
                    cur2[best] = channels[best]
            else:
                active -= 1

        offsets[n + 1] = pos

    return out_k1, out_k2, out_v, offsets


@njit(cache=True)
def loocv_train_acc(keys1, keys2, counts, offsets, y_codes, required_correct):
    """LOOCV 1NN over all bags, computing each pair intersection once.

    The histogram intersection is symmetric, so the full leave-one-out pass
    only needs the upper triangle of the similarity matrix. Rows are
    processed in order and the same early-abandon test as the sequential
    version is applied before each row: if required_correct can no longer
    be reached, processing stops.

    Returns (n_done, correct, preds): preds[i] is the index of the nearest
    neighbour of bag i for i < n_done (first maximum, skipping i itself).
    If the abandon test fired, n_done < n and correct is -1.
    """
    n = len(offsets) - 1
    sims = np.zeros((n, n), dtype=np.int32)
    preds = np.full(n, -1, dtype=np.int64)
    correct = 0

    for i in range(n):
        if correct + n - i < required_correct:
            return i, -1, preds

        a0, a1 = offsets[i], offsets[i + 1]
        for j in range(i + 1, n):
            sim = _histogram_intersection(
                keys1, keys2, counts, a0, a1, offsets[j], offsets[j + 1]
            )
            sims[i, j] = sim
            sims[j, i] = sim

        best_sim = -1
        nn = -1
        for j in range(n):
            if j == i:
                continue
            s = sims[i, j]
            if s > best_sim:
                best_sim = s
                nn = j

        preds[i] = nn
        if y_codes[nn] == y_codes[i]:
            correct += 1

    return n, correct, preds


@njit(cache=True, nogil=True)
def nn_first_max(sims):
    """First-maximum nearest neighbour per row, flagging tie events.

    Replicates the deterministic part of the sequential scan: nn0[r] is the
    index of the first maximum of row r, and has_tie[r] is True if any
    element equalled the running maximum during the scan (the only points
    where the random tie-break would consume a draw). Rows without tie
    events need no random draws, so their result is final.
    """
    n_rows, n_cols = sims.shape
    nn0 = np.empty(n_rows, dtype=np.int64)
    has_tie = np.zeros(n_rows, dtype=np.bool_)

    for r in range(n_rows):
        best = np.int64(-1)
        nn = -1
        for j in range(n_cols):
            s = sims[r, j]
            if s > best:
                best = s
                nn = j
            elif s == best:
                has_tie[r] = True
        nn0[r] = nn

    return nn0, has_tie


@njit(cache=True, nogil=True)
def nn_tie_break(sims, draws):
    """Nearest neighbour per row with precomputed tie-break draws.

    Replicates the sequential scan with a fresh, identically seeded random
    state per row: each row consumes draws from the start of ``draws``, one
    per tie event (an element equal to the running maximum), taking the tied
    index when the draw is below 0.5. With a seeded random state every row's
    generator yields the same sequence, so a single precomputed pool serves
    all rows exactly.
    """
    n_rows, n_cols = sims.shape
    nn = np.empty(n_rows, dtype=np.int64)

    for r in range(n_rows):
        best = np.int64(-1)
        chosen = -1
        d = 0
        for j in range(n_cols):
            s = sims[r, j]
            if s > best:
                best = s
                chosen = j
            elif s == best:
                if draws[d] < 0.5:
                    chosen = j
                d += 1
        nn[r] = chosen

    return nn


@njit(cache=True, nogil=True)
def nn_predict_loocv(keys1, keys2, counts, offsets, train_num):
    """Index of the 1NN of bag train_num among all other bags."""
    a0, a1 = offsets[train_num], offsets[train_num + 1]
    best_sim = -1
    nn = -1
    for n in range(len(offsets) - 1):
        if n == train_num:
            continue
        sim = _histogram_intersection(
            keys1, keys2, counts, a0, a1, offsets[n], offsets[n + 1]
        )
        if sim > best_sim:
            best_sim = sim
            nn = n
    return nn


@njit(cache=True, nogil=True)
def nn_similarities_all(
    keys1, keys2, counts, offsets, t_keys1, t_keys2, t_counts, t_offsets
):
    """Similarities of every test bag against every train bag.

    Returns an (n_test, n_train) int64 matrix in a single call, avoiding
    per-test-case call overhead.
    """
    n_train = len(offsets) - 1
    n_test = len(t_offsets) - 1
    sims = np.zeros((n_test, n_train), dtype=np.int64)
    for t in range(n_test):
        t0, t1 = t_offsets[t], t_offsets[t + 1]
        for m in range(n_train):
            b0, b1 = offsets[m], offsets[m + 1]
            sim = 0
            i, j = t0, b0
            while i < t1 and j < b1:
                ka1, ka2 = t_keys1[i], t_keys2[i]
                kb1, kb2 = keys1[j], keys2[j]
                if ka1 == kb1 and ka2 == kb2:
                    sim += min(t_counts[i], counts[j])
                    i += 1
                    j += 1
                elif ka1 < kb1 or (ka1 == kb1 and ka2 < kb2):
                    i += 1
                else:
                    j += 1
            sims[t, m] = sim
    return sims
