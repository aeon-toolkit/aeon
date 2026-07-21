"""TDE classifiers.

Dictionary based TDE classifiers based on SFA transform. Contains a single
IndividualTDE and TDE.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["TemporalDictionaryEnsemble", "IndividualTDE", "histogram_intersection"]

import math
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from numba import njit, types
from numba.typed import Dict
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.dictionary_based._tde_sfa import (
    _TDE_SFA,
    combine_channel_bags,
    loocv_train_acc,
    nn_first_max,
    nn_predict_loocv,
    nn_similarities_all,
    nn_tie_break,
)
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
            tde.fit(X_subsample, y_subsample)
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

                preds = clf.predict(X[oob])

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
