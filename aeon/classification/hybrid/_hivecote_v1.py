"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1.

Hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

__maintainer__ = []
__all__ = ["HIVECOTEV1"]

from aeon.classification.dictionary_based import ContractableBOSS

# import base
from aeon.classification.hybrid._base_hive_cote import BaseHIVECOTE
from aeon.classification.interval_based import (
    RandomIntervalSpectralEnsembleClassifier,
    TimeSeriesForestClassifier,
)
from aeon.classification.shapelet_based import ShapeletTransformClassifier


class HIVECOTEV1(BaseHIVECOTE):
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
        "capability:multivariate": False,
        "capability:contractable": False,
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
        self.parallel_backend = parallel_backend

        super().__init__(
            estimators=None,
            alpha=4,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

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
            self._stc_params = {"n_shapelet_samples": self._DEFAULT_N_SHAPELETS}
        else:
            self._stc_params = self.stc_params

        if self.tsf_params is None:
            self._tsf_params = {"n_estimators": self._DEFAULT_N_TREES}
        else:
            self._tsf_params = self.tsf_params

        if self.rise_params is None:
            self._rise_params = {"n_estimators": self._DEFAULT_N_TREES}
        else:
            self._rise_params = self.rise_params

        if self.cboss_params is None:
            self._cboss_params = {
                "n_parameter_samples": self._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": self._DEFAULT_MAX_ENSEMBLE_SIZE,
            }
        else:
            self._cboss_params = self.cboss_params

        self.estimators = [
            ("STC", ShapeletTransformClassifier(**self._stc_params)),
            ("TSF", TimeSeriesForestClassifier(**self._tsf_params)),
            ("RISE", RandomIntervalSpectralEnsembleClassifier(**self._rise_params)),
            ("cBOSS", ContractableBOSS(**self._cboss_params)),
        ]

        return super()._fit(X, y)

    @property
    def stc_weight_(self):
        return self.get_component_weights().get("STC", 0.0)

    @property
    def tsf_weight_(self):
        return self.get_component_weights().get("TSF", 0.0)

    @property
    def rise_weight_(self):
        return self.get_component_weights().get("RISE", 0.0)

    @property
    def cboss_weight_(self):
        return self.get_component_weights().get("cBOSS", 0.0)

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
