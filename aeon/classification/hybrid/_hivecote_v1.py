"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1.

Hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

__maintainer__ = []
__all__ = ["HIVECOTEV1"]

from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.dictionary_based import ContractableBOSS
from aeon.classification.interval_based import (
    RandomIntervalSpectralEnsembleClassifier,
    TimeSeriesForestClassifier,
)
from aeon.classification.shapelet_based import ShapeletTransformClassifier


class HIVECOTEV1(BaseClassifier):
    """
    Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1.

    An ensemble of the STC, TSF, RISE and cBOSS classifiers from different feature
    representations using the CAWPE structure as described in [1]_. The default
    implementation differs from the one described in [1]_, in that the STC component
    uses the out of bag error (OOB) estimates for weights (described in [2]_) rather
    than the cross-validation estimate. OOB is an order of magnitude faster and on
    average as good as CV. This means that this version of HIVE COTE is a bit faster
    than HC2, although less accurate on average.

    Parameters
    ----------
    stc_params : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters with a 2 hour transform contract.
    tsf_params : dict or None, default=None
        Parameters for the TimeSeriesForestClassifier module. If None, uses the default
        parameters with n_estimators set to 500.
    rise_params : dict or None, default=None
        Parameters for the RandomIntervalSpectralForest module. If None, uses the
        default parameters with n_estimators set to 500.
    cboss_params : dict or None, default=None
        Parameters for the ContractableBOSS module. If None, uses the default
        parameters.
    verbose : int, default=0
        Level of output printed to the console (for information only).
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib for Catch22,
        if None a 'prefer' value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.
    stc_weight_ : float
        The weight for STC probabilities.
    tsf_weight_ : float
        The weight for TSF probabilities.
    rise_weight_ : float
        The weight for RISE probabilities.
    cboss_weight_ : float
        The weight for cBOSS probabilities.

    See Also
    --------
    ShapeletTransformClassifier, TimeSeriesForestClassifier,
    RandomIntervalSpectralForest, ContractableBOSS
        All components of HIVECOTE.
    HIVECOTEV2
        Successor to HIVECOTEV1.

    Notes
    -----
    For the Java version, see
    `https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/HIVE_COTE.java`_.

    References
    ----------
    .. [1] Anthony Bagnall, Michael Flynn, James Large, Jason Lines and
       Matthew Middlehurst. "On the usage and performance of the Hierarchical Vote
       Collective of Transformation-based Ensembles version 1.0 (hive-cote v1.0)"
       International Workshop on Advanced Analytics and Learning on Temporal Data 2020
    .. [2] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." Machine Learning (2021).
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        stc_params=None,
        tsf_params=None,
        rise_params=None,
        cboss_params=None,
        verbose=0,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.stc_params = stc_params
        self.tsf_params = tsf_params
        self.rise_params = rise_params
        self.cboss_params = cboss_params
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        self.stc_weight_ = 0
        self.tsf_weight_ = 0
        self.rise_weight_ = 0
        self.cboss_weight_ = 0

        self._stc_params = stc_params
        self._tsf_params = tsf_params
        self._rise_params = rise_params
        self._cboss_params = cboss_params
        self._stc = None
        self._tsf = None
        self._rise = None
        self._cboss = None

        super().__init__()

    _DEFAULT_N_TREES = 500
    _DEFAULT_N_SHAPELETS = 10000
    _DEFAULT_N_PARA_SAMPLES = 250
    _DEFAULT_MAX_ENSEMBLE_SIZE = 50

    def _fit(self, X, y):
        """Fit HIVE-COTE 1.0 to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        y : array-like, shape = [n_cases]
            The class labels.

        Returns
        -------
        self :
            Reference to self.
        """
        if self.stc_params is None:
            self._stc_params = {"n_shapelet_samples": HIVECOTEV1._DEFAULT_N_SHAPELETS}
        if self.tsf_params is None:
            self._tsf_params = {"n_estimators": HIVECOTEV1._DEFAULT_N_TREES}
        if self.rise_params is None:
            self._rise_params = {"n_estimators": HIVECOTEV1._DEFAULT_N_TREES}
        if self.cboss_params is None:
            self._cboss_params = {
                "n_parameter_samples": HIVECOTEV1._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": HIVECOTEV1._DEFAULT_MAX_ENSEMBLE_SIZE,
            }

        # Build STC
        self._stc = ShapeletTransformClassifier(
            **self._stc_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        train_preds = self._stc.fit_predict(X, y)
        self.stc_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("STC ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("STC weight = " + str(self.stc_weight_))  # noqa

        # Build TSF
        self._tsf = TimeSeriesForestClassifier(
            **self._tsf_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        train_preds = self._tsf.fit_predict(X, y)
        self.tsf_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("TSF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("TSF weight = " + str(self.tsf_weight_))  # noqa

        # Build RISE
        self._rise = RandomIntervalSpectralEnsembleClassifier(
            **self._rise_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        train_preds = self._rise.fit_predict(X, y)
        self.rise_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("RISE ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("RISE weight = " + str(self.rise_weight_))  # noqa

        # Build cBOSS
        self._cboss = ContractableBOSS(
            **self._cboss_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        train_preds = self._cboss.fit_predict(X, y)
        self.cboss_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("cBOSS ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("cBOSS weight = " + str(self.cboss_weight_))  # noqa

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))

        # Call predict proba on each classifier, multiply the probabilities by the
        # classifiers weight then add them to the current HC1 probabilities
        dists = np.add(
            dists,
            self._stc.predict_proba(X) * (np.ones(self.n_classes_) * self.stc_weight_),
        )
        dists = np.add(
            dists,
            self._tsf.predict_proba(X) * (np.ones(self.n_classes_) * self.tsf_weight_),
        )
        dists = np.add(
            dists,
            self._rise.predict_proba(X)
            * (np.ones(self.n_classes_) * self.rise_weight_),
        )
        dists = np.add(
            dists,
            self._cboss.predict_proba(X)
            * (np.ones(self.n_classes_) * self.cboss_weight_),
        )

        # Make each instances probability array sum to 1 and return
        return dists / dists.sum(axis=1, keepdims=True)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            HIVECOTEV1 provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        from sklearn.ensemble import RandomForestClassifier

        if parameter_set == "results_comparison":
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                "tsf_params": {"n_estimators": 3},
                "rise_params": {"n_estimators": 3},
                "cboss_params": {"n_parameter_samples": 5, "max_ensemble_size": 3},
            }
        else:
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=1),
                    "n_shapelet_samples": 5,
                    "max_shapelets": 5,
                    "batch_size": 5,
                },
                "tsf_params": {"n_estimators": 1},
                "rise_params": {"n_estimators": 1},
                "cboss_params": {"n_parameter_samples": 1, "max_ensemble_size": 1},
            }
