# -*- coding: utf-8 -*-
"""WEASEL 2.0 classifier.

A Random Dilated Dictionary Transform for Fast, Accurate and Constrained Memory
Time Series Classification.

"""

__author__ = ["patrickzib"]
__all__ = ["WEASEL_V2", "WEASELTransformerV2"]

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFAFast

# some constants on input parameters for WEASEL v2
SWITCH_SMALL_INSTANCES = 250
SWITCH_MEDIUM_LENGTH = 100

ENSEMBLE_SIZE_SMALL = 50
ENSEMBLE_SIZE_MEDIUM = 100
ENSEMBLE_SIZE_LARGE = 150

MAX_WINDOW_SMALL = 24
MAX_WINDOW_MEDIUM = 44
MAX_WINDOW_LARGE = 84


class WEASEL_V2(BaseClassifier):
    """Word Extraction for Time Series Classification (WEASEL) v2.0.

    Overview: Input 'n' series length 'm'
    WEASEL is a dictionary classifier that builds a bag-of-patterns using SFA
    for different window lengths and learns a logistic regression classifier
    on this bag.

    WEASEL 2.0 has three key parameters that are automcatically set based on the
    length of the time series:
    (1) Minimal window length: Typically defaulted to 4
    (2) Maximal window length: Typically chosen from
        24, 44 or 84 depending on the time series length.
    (3) Ensemble size: Typically chosen from 50, 100, 150, to derive
        a feature vector of roughly 20𝑘 up to 70𝑘 features (distinct words).

    From the other parameters passed, WEASEL chosen random values for each set
    of configurations. E.g. for each of 150 configurations, a random value is chosen
    from the below options.

    Parameters
    ----------
    min_window : int, default=4,
        Minimal length of the subsequences to compute words from.
    norm_options : array of bool, default=[False],
        If the array contains True, words are computed over mean-normed TS
        If the array contains False, words are computed over raw TS
        If both are set, words are computed for both.
        A value will be randomly chosen for each parameter-configuration.
    word_lengths : array of int, default=[7, 8],
        Length of the words to compute. A value will be randomly chosen for each
        parameter-configuration.
    use_first_differences: array of bool, default=[True, False],
        If the array contains True, words are computed over first order differences.
        If the array contains False, words are computed over the raw time series.
        If both are set, words are computed for both.
    feature_selection: {"chi2_top_k", "none", "random"}, default: chi2_top_k
        Sets the feature selections strategy to be used. Large amounts of memory may be
        needed depending on the setting of bigrams (true is more) or
        alpha (larger is more).
        'chi2_top_k' reduces the number of words to at most 'max_feature_count',
        dropping values based on p-value.
        'random' reduces the number to at most 'max_feature_count',
        by randomly selecting features.
        'none' does not apply any feature selection and yields large bag of words
    max_feature_count : int, default=30_000
       size of the dictionary - number of words to use - if feature_selection set to
       "chi2" or "random". Else ignored.
    support_probabilities: bool, default: False
        If set to False, a RidgeClassifierCV will be trained, which has higher accuracy
        and is faster, yet does not support predict_proba.
        If set to True, a LogisticRegression will be trained, which does support
        predict_proba(), yet is slower and typically less accurate. predict_proba() is
        needed for example in Early-Classification like TEASER.
    random_state: int or None, default=None
        Seed for random, integer

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.

    See Also
    --------
    MUSE

    References
    ----------
    .. [1] Patrick Schäfer and Ulf Leser, "WEASEL 2.0 -- A Random Dilated Dictionary
    Transform for Fast, Accurate and Memory Constrained Time Series Classification",
    Preprint, https://arxiv.org/abs/2301.10194

    Examples
    --------
    >>> from sktime.classification.dictionary_based import WEASEL_V2
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = WEASEL_V2()
    >>> clf.fit(X_train, y_train)
    WEASEL_V2(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        min_window=4,
        norm_options=(False,),  # tuple
        word_lengths=(7, 8),
        use_first_differences=(True, False),
        feature_selection="chi2_top_k",
        max_feature_count=30_000,
        random_state=None,
        support_probabilities=False,
        n_jobs=4,
    ):
        self.norm_options = norm_options
        self.word_lengths = word_lengths

        self.random_state = random_state

        self.min_window = min_window

        self.max_feature_count = max_feature_count
        self.use_first_differences = use_first_differences
        self.feature_selection = feature_selection

        self.clf = None
        self.n_jobs = n_jobs
        self.support_probabilities = support_probabilities

        super(WEASEL_V2, self).__init__()

    def _fit(self, X, y):
        """Build a WEASEL classifiers from the training set (X, y).

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.
        """
        # Window length parameter space dependent on series length

        ...

        self.transform = WEASELTransformerV2(
            min_window=self.min_window,
            norm_options=self.norm_options,
            word_lengths=self.word_lengths,
            use_first_differences=self.use_first_differences,
            feature_selection=self.feature_selection,
            max_feature_count=self.max_feature_count,
            random_state=self.random_state,
            support_probabilities=self.support_probabilities,
            n_jobs=self.n_jobs,
        )
        words = self.transform.fit_transform(X, y)

        if not self.support_probabilities:
            self.clf = RidgeClassifierCV(alphas=np.logspace(-1, 5, 10))
        else:
            self.clf = LogisticRegression(
                max_iter=5000,
                solver="liblinear",
                dual=True,
                penalty="l2",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        self.clf.fit(words, y)

        if hasattr(self.clf, "best_score_"):
            self.cross_val_score = self.clf.best_score_

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        bag = self.transform.transform(X)
        return self.clf.predict(bag)

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        bag = self.transform.transform(X)
        if self.support_probabilities:
            return self.clf.predict_proba(bag)
        else:
            raise ValueError(
                "Error in WEASEL v2, please set support_probabilities=True, to"
                + "allow for probabilities to be computed."
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "feature_selection": "none",
            "support_probabilities": True,
        }


class WEASELTransformerV2:
    """The Word Extraction for Time Series Classifier v2.0 Transformation.

    WEASEL 2.0 has three key parameters that are automcatically set based on the
    length of the time series:
    (1) Minimal window length: Typically defaulted to 4
    (2) Maximal window length: Typically chosen from
        24, 44 or 84 depending on the time series length.
    (3) Ensemble size: Typically chosen from 50, 100, 150, to derive
        a feature vector of roughly 20𝑘 up to 70𝑘 features (distinct words).

    From the other parameters passed, WEASEL chosen random values for each set
    of configurations. E.g. for each of 150 configurations, a random value is chosen
    from the below options.

    Parameters
    ----------
    min_window : int, default=4,
        Minimal length of the subsequences to compute words from.
    norm_options : array of bool, default=[False],
        If the array contains True, words are computed over mean-normed TS
        If the array contains False, words are computed over raw TS
        If both are set, words are computed for both.
        A value will be randomly chosen for each parameter-configuration.
    word_lengths : array of int, default=[7, 8],
        Length of the words to compute. A value will be randomly chosen for each
        parameter-configuration.
    use_first_differences: array of bool, default=[True, False],
        If the array contains True, words are computed over first order differences.
        If the array contains False, words are computed over the raw time series.
        If both are set, words are computed for both.
    feature_selection: {"chi2_top_k", "none", "random"}, default: chi2_top_k
        Sets the feature selections strategy to be used. Large amounts of memory may be
        needed depending on the setting of bigrams (true is more) or
        alpha (larger is more).
        'chi2_top_k' reduces the number of words to at most 'max_feature_count',
        dropping values based on p-value.
        'random' reduces the number to at most 'max_feature_count',
        by randomly selecting features.
        'none' does not apply any feature selection and yields large bag of words
    max_feature_count : int, default=30_000
       size of the dictionary - number of words to use - if feature_selection set to
       "chi2" or "random". Else ignored.
    random_state: int or None, default=None
        Seed for random, integer
    """

    # _tags = {
    #     "univariate-only": True,
    #     "scitype:transform-input": "Series",
    #     # what is the scitype of X: Series, or Panel
    #     "scitype:transform-output": "Series",
    #     # what scitype is returned: Primitives, Series, Panel
    #     "scitype:instancewise": False,  # is this an instance-wise transform?
    #     "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
    #     "y_inner_mtype": "pd_Series_Table",  # which mtypes does y require?
    #     "requires_y": True,  # does y need to be passed in fit?
    # }

    def __init__(
        self,
        min_window=4,
        norm_options=(False,),  # tuple
        word_lengths=(7, 8),
        use_first_differences=(True, False),
        feature_selection="chi2_top_k",
        max_feature_count=30_000,
        random_state=None,
        support_probabilities=False,
        n_jobs=4,
    ):
        self.min_window = min_window
        self.norm_options = norm_options
        self.word_lengths = word_lengths
        self.use_first_differences = use_first_differences
        self.feature_selection = feature_selection
        self.max_feature_count = max_feature_count
        self.random_state = random_state
        self.support_probabilities = support_probabilities
        self.n_jobs = n_jobs

        self.alphabet_sizes = [2]
        self.binning_strategies = ["equi-depth", "equi-width"]

        self.anova = False
        self.variance = True
        self.bigrams = False
        self.lower_bounding = True
        self.remove_repeat_words = False

        self.max_window = MAX_WINDOW_LARGE
        self.ensemble_size = ENSEMBLE_SIZE_LARGE
        self.window_sizes = []
        self.series_length_ = 0
        self.n_instances_ = 0

        self.SFA_transformers = []

    def fit_transform(self, X, y=None):
        """Build a WEASEL model from the training set (X, y).

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        scipy csr_matrix, transformed features
        """
        # Window length parameter space dependent on series length
        self.n_instances_, self.series_length_ = X.shape[0], X.shape[-1]
        XX = X.squeeze(1)

        # avoid overfitting with too many features
        if self.n_instances_ < SWITCH_SMALL_INSTANCES:
            self.max_window = MAX_WINDOW_SMALL
            self.ensemble_size = ENSEMBLE_SIZE_SMALL
        elif self.series_length_ < SWITCH_MEDIUM_LENGTH:
            self.max_window = MAX_WINDOW_MEDIUM
            self.ensemble_size = ENSEMBLE_SIZE_MEDIUM
        else:
            self.max_window = MAX_WINDOW_LARGE
            self.ensemble_size = ENSEMBLE_SIZE_LARGE

        self.max_window = int(min(self.series_length_, self.max_window))
        if self.min_window > self.max_window:
            raise ValueError(
                f"Error in WEASEL, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={self.max_window},"
                f" series length is {self.series_length_}"
                f" try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )

        # Randomly choose window sizes
        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, backend="threading")(
            delayed(_parallel_fit)(
                i,
                XX,
                y.copy(),
                self.window_sizes,
                self.alphabet_sizes,
                self.word_lengths,
                self.series_length_,
                self.norm_options,
                self.use_first_differences,
                self.binning_strategies,
                self.variance,
                self.anova,
                self.bigrams,
                self.lower_bounding,
                self.n_jobs,
                self.max_feature_count,
                self.ensemble_size,
                self.feature_selection,
                self.remove_repeat_words,
                self.random_state,
            )
            for i in range(self.ensemble_size)
        )

        sfa_words = []
        for words, transformer in parallel_res:
            self.SFA_transformers.extend(transformer)
            sfa_words.extend(words)

        # merging arrays from different threads
        if type(sfa_words[0]) is np.ndarray:
            all_words = np.concatenate(sfa_words, axis=1)
        else:
            all_words = hstack(sfa_words)

        self.total_features_count = all_words.shape[1]

        return all_words

    def transform(self, X, y=None):
        """Transform X into a WEASEL model.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
           The data to make predictions for.
        y : ignored argument for interface compatibility

        Returns
        -------
        scipy csr_matrix, transformed features
        """
        return self._transform_words(X)

    def _transform_words(self, X):
        XX = X.squeeze(1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, backend="threading")(
            delayed(transformer.transform)(XX) for transformer in self.SFA_transformers
        )

        all_words = []
        for words in parallel_res:
            all_words.append(words)

        if type(all_words[0]) is np.ndarray:
            all_words = np.concatenate(all_words, axis=1)
        else:
            all_words = hstack(all_words)

        return all_words


def _parallel_fit(
    i,
    X,
    y,
    window_sizes,
    alphabet_sizes,
    word_lengths,
    series_length,
    norm_options,
    use_first_differences,
    binning_strategies,
    variance,
    anova,
    bigrams,
    lower_bounding,
    n_jobs,
    max_feature_count,
    ensemble_size,
    feature_selection,
    remove_repeat_words,
    random_state,
):
    if random_state is None:
        rng = check_random_state(None)
    else:
        rng = check_random_state(random_state + i)

    window_size = rng.choice(window_sizes)
    dilation = np.maximum(
        1,
        np.int32(2 ** rng.uniform(0, np.log2((series_length - 1) / (window_size - 1)))),
    )

    alphabet_size = rng.choice(alphabet_sizes)

    # maximize word-length
    word_length = min(window_size - 2, rng.choice(word_lengths))
    norm = rng.choice(norm_options)
    binning_strategy = rng.choice(binning_strategies)

    all_transformers = []
    all_words = []
    for first_difference in use_first_differences:
        transformer = SFAFast(
            variance=variance,
            word_length=word_length,
            alphabet_size=alphabet_size,
            window_size=window_size,
            norm=norm,
            anova=anova,
            binning_method=binning_strategy,
            remove_repeat_words=remove_repeat_words,
            bigrams=bigrams,
            dilation=dilation,
            lower_bounding=lower_bounding,
            first_difference=first_difference,
            feature_selection=feature_selection,
            max_feature_count=max_feature_count // ensemble_size,
            random_state=i,
            return_sparse=False,
            n_jobs=n_jobs,
        )

        # generate SFA words on sample
        words = transformer.fit_transform(X, y)
        all_words.append(words)
        all_transformers.append(transformer)
    return all_words, all_transformers
