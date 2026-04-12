"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

Upgraded hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

__maintainer__ = ["MatthewMiddlehurst", "TonyBagnall"]
__all__ = ["HIVECOTEV2"]

import numpy as np

from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.hybrid._base_hive_cote import BaseHIVECOTE
from aeon.classification.interval_based._drcif import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier


class HIVECOTEV2(BaseHIVECOTE):
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

    _DEFAULT_N_TREES = 500
    _DEFAULT_N_SHAPELETS = 10000
    _DEFAULT_N_KERNELS = 2000
    _DEFAULT_N_ESTIMATORS = 25
    _DEFAULT_N_PARA_SAMPLES = 250
    _DEFAULT_MAX_ENSEMBLE_SIZE = 50
    _DEFAULT_RAND_PARAMS = 50

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
        self.parallel_backend = parallel_backend

        super().__init__(
            estimators=None,
            alpha=4,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

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
        _stc_params = self.stc_params or {
            "n_shapelet_samples": self._DEFAULT_N_SHAPELETS
        }
        _drcif_params = self.drcif_params or {"n_estimators": self._DEFAULT_N_TREES}
        _arsenal_params = self.arsenal_params or {
            "n_kernels": self._DEFAULT_N_KERNELS,
            "n_estimators": self._DEFAULT_N_ESTIMATORS,
        }
        _tde_params = self.tde_params or {
            "n_parameter_samples": self._DEFAULT_N_PARA_SAMPLES,
            "max_ensemble_size": self._DEFAULT_MAX_ENSEMBLE_SIZE,
            "randomly_selected_params": self._DEFAULT_RAND_PARAMS,
        }

        # If we are contracting split the contract time between each algorithm
        if self.time_limit_in_minutes > 0:
            ct = self.time_limit_in_minutes / 6
            _stc_params["time_limit_in_minutes"] = ct
            _drcif_params["time_limit_in_minutes"] = ct
            _arsenal_params["time_limit_in_minutes"] = ct
            _tde_params["time_limit_in_minutes"] = ct

        # Build STC
        # import from _base_hive_cote.py
        self.estimators = [
            ("STC", ShapeletTransformClassifier(**_stc_params)),
            ("DrCIF", DrCIFClassifier(**_drcif_params)),
            ("Arsenal", Arsenal(**_arsenal_params)),
            ("TDE", TemporalDictionaryEnsemble(**_tde_params)),
        ]

        # 4. 把剩下所有脏活累活（训练、算权重等）全部丢给父类！
        return super()._fit(X, y)

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
        dists = super()._predict_proba(X)

        if self.save_component_probas:
            self.component_probas = {
                name: est.predict_proba(X)
                for name, est in zip(self.component_names_, self.fitted_estimators_)
            }
        # Make each instances probability array sum to 1 and return
        return dists

    @property
    def stc_weight_(self):
        return self.get_component_weights().get("STC", 0.0)

    @property
    def drcif_weight_(self):
        return self.get_component_weights().get("DrCIF", 0.0)

    @property
    def arsenal_weight_(self):
        return self.get_component_weights().get("Arsenal", 0.0)

    @property
    def tde_weight_(self):
        return self.get_component_weights().get("TDE", 0.0)

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
