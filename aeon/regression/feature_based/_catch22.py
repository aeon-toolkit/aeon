# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Catch22 Regressor.

Pipeline regressor using the Catch22 transformer and an estimator.
"""

__author__ = ["MatthewMiddlehurst", "RavenRudi", "TonyBagnall"]
__all__ = ["Catch22Regressor"]

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from aeon.base._base import _clone_estimator
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection.catch22 import Catch22


class Catch22Regressor(BaseRegressor):
    """Canonical Time-series Characteristics (catch22) regressor.

    This regressor simply transforms the input data using the Catch22 [1]
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    features : int/str or List of int/str, optional, default="all"
        The Catch22 features to extract by feature index, feature name as a str or as a
        list of names or indices for multiple features. If "all", all features are
        extracted.
        Valid features are as follows:
            ["DN_HistogramMode_5", "DN_HistogramMode_10",
            "SB_BinaryStats_diff_longstretch0", "DN_OutlierInclude_p_001_mdrmd",
            "DN_OutlierInclude_n_001_mdrmd", "CO_f1ecac", "CO_FirstMin_ac",
            "SP_Summaries_welch_rect_area_5_1", "SP_Summaries_welch_rect_centroid",
            "FC_LocalSimple_mean3_stderr", "CO_trev_1_num", "CO_HistogramAMI_even_2_5",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi", "MD_hrv_classic_pnn40",
            "SB_BinaryStats_mean_longstretch1", "SB_MotifThree_quantile_hh",
            "FC_LocalSimple_mean1_tauresrat", "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SB_TransitionMatrix_3ac_sumdiagcov", "PD_PeriodicityWang_th0_01"]
    catch24 : bool, optional, default=True
        Extract the mean and standard deviation as well as the 22 Catch22 features if
        True. If a List of specific features to extract is provided, "Mean" and/or
        "StandardDeviation" must be added to the List to extract these features.
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, optional, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    use_pycatch22 : bool, optional, default=False
        Wraps the C based pycatch22 implementation for aeon.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
    estimator : sklearn regressor, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestRegressor(n_estimators=200)
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

    See Also
    --------
    Catch22

    References
    ----------
    .. [1] Lubba, Carl H., et al. "catch22: Canonical time-series characteristics."
        Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.
        https://link.springer.com/article/10.1007/s10618-019-00647-x

    Examples
    --------
    >>> from aeon.regression.feature_based import Catch22Regressor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from aeon.datasets import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, regression_target=True,
    ...                              random_state=0)
    >>> reg = Catch22Regressor(
    ...     estimator=RandomForestRegressor(n_estimators=5),
    ...     outlier_norm=True,
    ...     random_state=0,
    ... )
    >>> reg.fit(X, y)
    Catch22Regressor(...)
    >>> reg.predict(X)
    array([0.66497445, 1.52167747, 0.73353397, 1.57550709, 0.46036267,
           0.6494623 , 1.08156127, 1.09927538, 1.46025772, 0.37711294])
    """

    _tags = {
        "X_inner_mtype": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        features="all",
        catch24=True,
        outlier_norm=False,
        replace_nans=True,
        use_pycatch22=False,
        estimator=None,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.use_pycatch22 = use_pycatch22
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super(Catch22Regressor, self).__init__()

    def _fit(self, X, y):
        """Fit Catch22Regressor to training data.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape (n_instances, n_channels, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_instances], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i
        y : 1D np.array, of shape [n_instances] - target labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        self._transformer = Catch22(
            features=self.features,
            catch24=self.catch24,
            outlier_norm=self.outlier_norm,
            replace_nans=self.replace_nans,
            use_pycatch22=self.use_pycatch22,
            n_jobs=self._threads_to_use,
            parallel_backend=self.parallel_backend,
        )

        self._estimator = _clone_estimator(
            RandomForestRegressor(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._threads_to_use

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape (n_instances, n_channels, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_instances], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted target labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Catch22Regressor provides the following special sets:
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
            return {
                "estimator": RandomForestRegressor(n_estimators=10),
                "outlier_norm": True,
            }
        else:
            return {
                "estimator": RandomForestRegressor(n_estimators=2),
                "features": (
                    "Mean",
                    "DN_HistogramMode_5",
                    "SB_BinaryStats_mean_longstretch1",
                ),
            }
