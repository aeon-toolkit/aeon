# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""CIF classifier.

Interval based CIF classifier extracting catch22 features from random intervals.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["CanonicalIntervalForestClassifier"]

import numpy as np

from aeon.classification import BaseClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.series_as_features.interval_based import BaseIntervalForest
from aeon.transformations.collection import Catch22
from aeon.utils.numba.stats import row_mean, row_slope, row_std


class CanonicalIntervalForestClassifier(BaseIntervalForest, BaseClassifier):
    """Canonical Interval Forest Classifier (CIF).

    todo

    Implementation of the nterval based forest making use of the catch22 feature set
    on randomly selected intervals described in Middlehurst et al. (2020). [1]_

    Overview: Input "n" series with "d" dimensions of length "m".
    For each tree
        - Sample n_intervals intervals of random position and length
        - Subsample att_subsample_size catch22 or summary statistic attributes randomly
        - Randomly select dimension for each interval
        - Calculate attributes for each interval, concatenate to form new
          data set
        - Build decision tree on new data set
    ensemble the trees with averaged probability estimates

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_intervals : int or None, default=None
        Number of intervals to extract per tree, if None extracts
        (sqrt(series_length) * sqrt(n_dims)) intervals.
    att_subsample_size : int, default=8
        Number of catch22 or summary statistic attributes to subsample per tree.
    min_interval : int, default=3
        Minimum length of an interval.
    max_interval : int or None, default=None
        Maximum length of an interval, if None set to (series_length / 2).
    base_estimator : BaseEstimator or str, default="DTC"
        Base estimator for the ensemble, can be supplied a sklearn BaseEstimator or a
        string for suggested options.
        "DTC" uses the sklearn DecisionTreeClassifier using entropy as a splitting
        measure.
        "CIT" uses the aeon ContinuousIntervalTree, an implementation of the original
        tree used with embedded attribute processing for faster predictions.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of ndarray with shape (n_intervals,2)
        Stores indexes of each intervals start and end points for all classifiers.
    atts_ : list of shape (n_estimators) of array with shape (att_subsample_size)
        Attribute indexes of the subsampled catch22 or summary statistic for all
        classifiers.
    dims_ : list of shape (n_estimators) of array with shape (n_intervals)
        The dimension to extract attributes from each interval for all classifiers.

    See Also
    --------
    DrCIF

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/CIF.java>`_.

    References
    ----------
    .. [1] Matthew Middlehurst and James Large and Anthony Bagnall. "The Canonical
       Interval Forest (CIF) Classifier for Time Series Classification."
       IEEE International Conference on Big Data 2020

    Examples
    --------
    >>> from aeon.classification.interval_based import CanonicalIntervalForestClassifier
    >>> from aeon.datasets import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = CanonicalIntervalForestClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    CanonicalIntervalForestClassifier(n_estimators=10, random_state=0)
    >>> clf.predict(X)
    array([0 1 0 1 0 0 1 1 1 0])
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
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        att_subsample_size=8,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pycatch22=False,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.use_pycatch22 = use_pycatch22
        if use_pycatch22:
            self.set_tags(**{"python_dependencies": "pycatch22"})

        if isinstance(base_estimator, ContinuousIntervalTree):
            replace_nan = "nan"
        else:
            replace_nan = 0

        interval_features = [
            Catch22(outlier_norm=True, use_pycatch22=use_pycatch22),
            row_mean,
            row_std,
            row_slope,
        ]

        super(CanonicalIntervalForestClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=None,
            att_subsample_size=att_subsample_size,
            replace_nan=replace_nan,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "n_estimators": 2,
            "n_intervals": 2,
            "att_subsample_size": 2,
        }
