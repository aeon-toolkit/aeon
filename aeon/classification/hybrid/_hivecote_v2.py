"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

Hybrid ensemble combining classifiers from four time-series representations using the
weighted probabilistic CAWPE structure.
"""

__maintainer__ = ["MatthewMiddlehurst", "TonyBagnall"]
__all__ = ["HIVECOTEV2"]

import warnings

import numpy as np

from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.hybrid._base_hive_cote import _BaseHIVECOTE
from aeon.classification.interval_based._drcif import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier


class HIVECOTEV2(_BaseHIVECOTE):
    """Hierarchical Vote Collective of Transformation Ensembles (HIVE-COTE) V2.

    HIVE-COTE 2.0 combines classifiers built on shapelet, interval, convolution, and
    dictionary representations. The STC, DrCIF, Arsenal, and TDE components are
    weighted by their training accuracy using the CAWPE structure described in [1]_.

    Parameters
    ----------
    stc_params : dict or None, default=None
        Parameters passed to ``ShapeletTransformClassifier``. If None, use
        ``n_shapelet_samples=10000`` and the component defaults.
    drcif_params : dict or None, default=None
        Parameters passed to ``DrCIFClassifier``. If None, use ``n_estimators=500``
        and the component defaults.
    arsenal_params : dict or None, default=None
        Parameters passed to ``Arsenal``. If None, use ``n_kernels=2000`` and
        ``n_estimators=25`` with the component defaults.
    tde_params : dict or None, default=None
        Parameters passed to ``TemporalDictionaryEnsemble``. If None, use
        ``n_parameter_samples=250``, ``max_ensemble_size=50``, and
        ``randomly_selected_params=50`` with the component defaults.
    time_limit_in_minutes : float, default=0
        Time contract for fitting the ensemble, in minutes. A positive value allocates
        one sixth of the contract to each component; otherwise, each component uses its
        configured estimator or parameter-sample count.
    save_component_probas : bool, default=False
        Whether ``predict`` and ``predict_proba`` save each component's probability
        predictions in ``component_probas``.

        .. deprecated::
            ``save_component_probas`` is deprecated because it mutates object
            state during prediction. Use ``predict_proba_with_components(X)``
            instead, which returns per-component probabilities without
            modifying the fitted estimator.
    verbose : int, default=0
        Level of output printed during fit. Level 1 reports HC2 progress, level 2 also
        reports component parameters, level 3 enables summary progress within each
        component, and level 4 and above enables detailed component progress.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Joblib parallel backend passed to the DrCIF component. If None, DrCIF uses its
        default backend. Valid options include ``"loky"``, ``"multiprocessing"``,
        ``"threading"``, or a custom backend.
    alpha : int or float, default=4
        Exponent applied to each component's training accuracy to calculate its
        CAWPE weight.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : np.ndarray of shape (n_classes_)
        The unique class labels.
    fitted_estimators_ : list of BaseClassifier
        The fitted STC, DrCIF, Arsenal, and TDE components, in that order.
    component_names_ : list of str
        Names corresponding to the estimators in ``fitted_estimators_``.
    weights_ : list of float
        CAWPE weights for the fitted components.
    stc_weight_ : float
        The weight for STC probabilities.
    drcif_weight_ : float
        The weight for DrCIF probabilities.
    arsenal_weight_ : float
        The weight for Arsenal probabilities.
    tde_weight_ : float
        The weight for TDE probabilities.
    component_probas : dict
        Component probability predictions from the most recent prediction when
        ``save_component_probas=True``. This deprecated attribute is created only when
        component probabilities are saved.

    See Also
    --------
    HIVECOTEV1
        The first version of HIVE-COTE.
    ShapeletTransformClassifier, DrCIFClassifier, Arsenal, TemporalDictionaryEnsemble
        The four HIVE-COTE 2.0 components.

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
    _verbose_name = "HC2"

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
        alpha=4,
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
            alpha=alpha,
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
        if self.stc_params is not None:
            self._stc_params = self.stc_params.copy()
        else:
            self._stc_params = {"n_shapelet_samples": self._DEFAULT_N_SHAPELETS}

        if self.drcif_params is not None:
            self._drcif_params = self.drcif_params.copy()
        else:
            self._drcif_params = {"n_estimators": self._DEFAULT_N_TREES}

        if self.arsenal_params is not None:
            self._arsenal_params = self.arsenal_params.copy()
        else:
            self._arsenal_params = {
                "n_kernels": self._DEFAULT_N_KERNELS,
                "n_estimators": self._DEFAULT_N_ESTIMATORS,
            }

        if self.tde_params is not None:
            self._tde_params = self.tde_params.copy()
        else:
            self._tde_params = {
                "n_parameter_samples": self._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": self._DEFAULT_MAX_ENSEMBLE_SIZE,
                "randomly_selected_params": self._DEFAULT_RAND_PARAMS,
            }

        # If we are contracting split the contract time between each algorithm
        if self.time_limit_in_minutes > 0:
            ct = self.time_limit_in_minutes / 6
            self._stc_params["time_limit_in_minutes"] = ct
            self._drcif_params["time_limit_in_minutes"] = ct
            self._arsenal_params["time_limit_in_minutes"] = ct
            self._tde_params["time_limit_in_minutes"] = ct

        # Build component estimators (stored in _estimators to avoid mutating
        # the self.estimators init parameter, for scikit-learn compatibility)
        drcif_build_params = self._drcif_params.copy()
        drcif_build_params.setdefault("parallel_backend", self.parallel_backend)
        self._estimators = [
            ("STC", ShapeletTransformClassifier(**self._stc_params)),
            ("DrCIF", DrCIFClassifier(**drcif_build_params)),
            ("Arsenal", Arsenal(**self._arsenal_params)),
            ("TDE", TemporalDictionaryEnsemble(**self._tde_params)),
        ]

        return super()._fit(X, y)

    def _log_fit_configuration(self):
        """Log the HC2 contract allocation when it is active."""
        if self.time_limit_in_minutes > 0:
            component_time = self.time_limit_in_minutes / 6
            self._log(
                f"[HC2] Contract time = {self.time_limit_in_minutes} minutes, "
                f"per-component allocation = {component_time:.4f} minutes"
            )

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        if self.save_component_probas:
            warnings.warn(
                "save_component_probas is deprecated because it mutates "
                "object state during prediction. Use "
                "predict_proba_with_components(X) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_probas, component_probas = self.predict_proba_with_components(X)
            self.component_probas = component_probas
            return final_probas

        return super()._predict_proba(X)

    def predict_proba_with_components(self, X):
        """Predict class probabilities and return per-component probabilities.

        Returns both the final ensemble probabilities and a dictionary mapping
        each component name to its individual probability predictions, without
        modifying object state.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        final_probas : np.ndarray of shape (n_cases, n_classes)
            The CAWPE-weighted ensemble probability predictions.
        component_probas : dict
            Dictionary mapping each component name to its probability predictions
            of shape (n_cases, n_classes).
        """
        self._check_is_fitted()

        component_probas = {
            name: est.predict_proba(X)
            for name, est in zip(self.component_names_, self.fitted_estimators_)
        }

        dists = np.zeros((X.shape[0], self.n_classes_))
        for i, name in enumerate(self.component_names_):
            dists = np.add(dists, component_probas[name] * self.weights_[i])

        final_probas = self._normalise_probabilities(dists)

        return final_probas, component_probas

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
