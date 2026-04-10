"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

Upgraded hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

__maintainer__ = ["MatthewMiddlehurst", "TonyBagnall"]
__all__ = ["HIVECOTEV2"]

from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.interval_based._drcif import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.utils.validation import check_n_jobs


class HIVECOTEV2(BaseClassifier):
    """
    Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

    An ensemble of the STC, DrCIF, Arsenal and TDE classifiers from different feature
    representations using the CAWPE structure as described in [1]_.

    Parameters
    ----------
    stc_params : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters with a 2 hour transform contract.
    drcif_params : dict or None, default=None
        Parameters for the DrCIF module. If None, uses the default parameters with
        n_estimators set to 500.
    arsenal_params : dict or None, default=None
        Parameters for the Arsenal module. If None, uses the default parameters.
    tde_params : dict or None, default=None
        Parameters for the TemporalDictionaryEnsemble module. If None, uses the default
        parameters.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding
        n_estimators/n_parameter_samples for each component.
        Default of 0 means n_estimators/n_parameter_samples for each component is used.
    save_component_probas : bool, default=False
        When predict/predict_proba is called, save each HIVE-COTEV2 component
        probability predictions in component_probas.
    verbose : int, default=0
        Level of output printed to the console
        - 0: no output
        - 1: HC2 level progress
        - 2: also print component parameter summaries
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
    drcif_weight_ : float
        The weight for DrCIF probabilities.
    arsenal_weight_ : float
        The weight for Arsenal probabilities.
    tde_weight_ : float
        The weight for TDE probabilities.
    component_probas : dict
        Only used if save_component_probas is true. Saved probability predictions for
        each HIVE-COTEV2 component.

    See Also
    --------
    HIVECOTEV1, ShapeletTransformClassifier, DrCIF, Arsenal, TemporalDictionaryEnsemble
        Components of HIVECOTE.

    Notes
    -----
    For the Java version, see
    `https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/HIVE_COTE.java`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." Machine Learning (2021).
    """

    _tags = {
        "capability:multivariate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        save_component_probas=False,
        verbose=0,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.stc_params = stc_params
        self.drcif_params = drcif_params
        self.arsenal_params = arsenal_params
        self.tde_params = tde_params
        self.time_limit_in_minutes = time_limit_in_minutes
        self.save_component_probas = save_component_probas
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        self.stc_weight_ = 0
        self.drcif_weight_ = 0
        self.arsenal_weight_ = 0
        self.tde_weight_ = 0
        self.component_probas = {}

        self._stc_params = stc_params
        self._drcif_params = drcif_params
        self._arsenal_params = arsenal_params
        self._tde_params = tde_params
        self._stc = None
        self._drcif = None
        self._arsenal = None
        self._tde = None

        super().__init__()

    _DEFAULT_N_TREES = 500
    _DEFAULT_N_SHAPELETS = 10000
    _DEFAULT_N_KERNELS = 2000
    _DEFAULT_N_ESTIMATORS = 25
    _DEFAULT_N_PARA_SAMPLES = 250
    _DEFAULT_MAX_ENSEMBLE_SIZE = 50
    _DEFAULT_RAND_PARAMS = 50

    def _fit(self, X, y):
        """Fit HIVE-COTE 2.0 to training data.

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

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self._n_jobs = check_n_jobs(self.n_jobs)
        total_start = perf_counter()

        self._log(
            f"[HC2] Starting fit: n_cases={X.shape[0]}, "
            f"n_channels={X.shape[1]}, n_timepoints={X.shape[2]}, "
            f"n_jobs={self._n_jobs}",
        )
        self._initialise_component_params()

        # Build STC
        self._stc, self.stc_weight_, stc_train_acc_ = self._fit_component(
            "STC",
            ShapeletTransformClassifier,
            self._stc_params,
            X,
            y,
        )
        # Build DrCIF
        self._drcif, self.drcif_weight_, drcif_train_acc_ = self._fit_component(
            "DrCIF",
            DrCIFClassifier,
            self._drcif_params,
            X,
            y,
        )
        # Build Arsenal
        self._arsenal, self.arsenal_weight_, arsenal_train_acc_ = self._fit_component(
            "Arsenal",
            Arsenal,
            self._arsenal_params,
            X,
            y,
        )
        # Build TDE
        self._tde, self.tde_weight_, tde_train_acc_ = self._fit_component(
            "TDE",
            TemporalDictionaryEnsemble,
            self._tde_params,
            X,
            y,
        )

        total_elapsed = perf_counter() - total_start
        self._log(f"[HC2] Finished fit in {total_elapsed:.2f}s")
        self._log(
            "[HC2] Component summary: "
            f"STC(train_acc={stc_train_acc_:.4f}, weight={self.stc_weight_:.4f}), "
            f"DrCIF(train_acc={drcif_train_acc_:.4f}, weight={self.drcif_weight_:.4f}),"
            f"Arsenal(train_acc={arsenal_train_acc_:.4f}, weight"
            f"={self.arsenal_weight_:.4f}), "
            f"TDE(train_acc={tde_train_acc_:.4f}, weight={self.tde_weight_:.4f})",
        )
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

    def _predict_proba(self, X, return_component_probas=False) -> np.ndarray:
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
        # classifiers weight then add them to the current HC2 probabilities
        stc_probas = self._stc.predict_proba(X)
        dists = np.add(
            dists,
            stc_probas * (np.ones(self.n_classes_) * self.stc_weight_),
        )
        drcif_probas = self._drcif.predict_proba(X)
        dists = np.add(
            dists,
            drcif_probas * (np.ones(self.n_classes_) * self.drcif_weight_),
        )
        arsenal_probas = self._arsenal.predict_proba(X)
        dists = np.add(
            dists,
            arsenal_probas * (np.ones(self.n_classes_) * self.arsenal_weight_),
        )
        tde_probas = self._tde.predict_proba(X)
        dists = np.add(
            dists,
            tde_probas * (np.ones(self.n_classes_) * self.tde_weight_),
        )

        if self.save_component_probas:
            self.component_probas = {
                "STC": stc_probas,
                "DrCIF": drcif_probas,
                "Arsenal": arsenal_probas,
                "TDE": tde_probas,
            }

        # Make each instances probability array sum to 1 and return
        return dists / dists.sum(axis=1, keepdims=True)

    def _get_component_params(self, params, default_params):
        """Return a working parameter dict for a HC2 component."""
        return default_params.copy() if params is None else params.copy()

    def _initialise_component_params(self):
        """Initialise working parameter dictionaries for HC2 components."""
        self._stc_params = self._get_component_params(
            self.stc_params,
            {"n_shapelet_samples": HIVECOTEV2._DEFAULT_N_SHAPELETS},
        )
        self._drcif_params = self._get_component_params(
            self.drcif_params,
            {"n_estimators": HIVECOTEV2._DEFAULT_N_TREES},
        )
        self._arsenal_params = self._get_component_params(
            self.arsenal_params,
            {
                "n_kernels": HIVECOTEV2._DEFAULT_N_KERNELS,
                "n_estimators": HIVECOTEV2._DEFAULT_N_ESTIMATORS,
            },
        )
        self._tde_params = self._get_component_params(
            self.tde_params,
            {
                "n_parameter_samples": HIVECOTEV2._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": HIVECOTEV2._DEFAULT_MAX_ENSEMBLE_SIZE,
                "randomly_selected_params": HIVECOTEV2._DEFAULT_RAND_PARAMS,
            },
        )

        if self.time_limit_in_minutes > 0:
            ct = self.time_limit_in_minutes / 6
            self._log(
                f"[HC2] Contract time = {self.time_limit_in_minutes} minutes, "
                f"per-component allocation = {ct:.4f} minutes",
                level=1,
            )
            self._stc_params["time_limit_in_minutes"] = ct
            self._drcif_params["time_limit_in_minutes"] = ct
            self._arsenal_params["time_limit_in_minutes"] = ct
            self._tde_params["time_limit_in_minutes"] = ct

    def _log(self, message, level=1):
        """Print a verbose message if the configured verbosity is high enough."""
        if self.verbose >= level:
            print(message, flush=True)  # noqa

    def _fit_component(self, name, estimator_cls, params, X, y):
        """Fit a single HC2 component."""
        build_params = params.copy()
        build_params.setdefault("random_state", self.random_state)
        build_params.setdefault("n_jobs", self._n_jobs)

        if estimator_cls is DrCIFClassifier:
            build_params.setdefault("parallel_backend", self.parallel_backend)

        self._log(f"[HC2] Starting {name}...", level=1)
        if self.verbose >= 2:
            self._log(f"[HC2] {name} params: {build_params}", level=2)

        start = perf_counter()

        estimator = estimator_cls(**build_params)
        train_preds = estimator.fit_predict(X, y)

        train_acc = accuracy_score(y, train_preds)
        weight = train_acc**4
        elapsed = perf_counter() - start

        self._log(
            f"[HC2] Finished {name} in {elapsed:.2f}s, "
            f"train_acc={train_acc:.4f}, weight={weight:.4f}",
            level=1,
        )

        return estimator, weight, train_acc

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            HIVECOTEV2 provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contacting
                    functionality

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        from sklearn.ensemble import RandomForestClassifier

        from aeon.classification.sklearn import RotationForestClassifier

        if parameter_set == "results_comparison":
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                "drcif_params": {
                    "n_estimators": 3,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                "arsenal_params": {"n_kernels": 50, "n_estimators": 3},
                "tde_params": {
                    "n_parameter_samples": 5,
                    "max_ensemble_size": 3,
                    "randomly_selected_params": 3,
                },
            }
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "stc_params": {
                    "estimator": RotationForestClassifier(contract_max_n_estimators=1),
                    "contract_max_n_shapelet_samples": 5,
                    "max_shapelets": 5,
                    "batch_size": 5,
                },
                "drcif_params": {
                    "contract_max_n_estimators": 1,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                "arsenal_params": {"n_kernels": 5, "contract_max_n_estimators": 1},
                "tde_params": {
                    "contract_max_n_parameter_samples": 1,
                    "max_ensemble_size": 1,
                    "randomly_selected_params": 1,
                },
            }
        else:
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=1),
                    "n_shapelet_samples": 5,
                    "max_shapelets": 5,
                    "batch_size": 5,
                },
                "drcif_params": {
                    "n_estimators": 1,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                "arsenal_params": {"n_kernels": 5, "n_estimators": 1},
                "tde_params": {
                    "n_parameter_samples": 1,
                    "max_ensemble_size": 1,
                    "randomly_selected_params": 1,
                },
            }
