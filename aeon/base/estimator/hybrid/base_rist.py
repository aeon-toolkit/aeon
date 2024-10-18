"""Base class for the RIST pipeline."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble._base import _set_random_states
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.transformations.collection import (
    ARCoefficientTransformer,
    PeriodogramTransformer,
)
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.interval_based import RandomIntervals
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)
from aeon.utils.numba.general import first_order_differences_3d
from aeon.utils.numba.stats import (
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_ppv,
    row_slope,
    row_std,
)
from aeon.utils.validation import check_n_jobs


class BaseRIST(ABC):
    """Randomised Interval-Shapelet Transformation (RIST) pipeline base.

    RIST is a hybrid pipeline using the RandomIntervalTransformer using
    Catch22 features and summary stats, and the RandomDilatedShapeletTransformer.
    Both transforms extract features from different series transformations (1st Order
    Differences, PeriodogramTransformer, and ARCoefficientTransformer).

    Parameters
    ----------
    n_intervals : int, callable or None, default=None,
        The number of intervals of random length, position and dimension to be
        extracted for the interval portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of intervals per `series_transformer` output.
        If None, extracts `int(np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5)`
        intervals where `Xt` is the series representation data.
    n_shapelets : int, callable or None, default=None,
        The number of shapelets of random dilation and position to be extracted for the
        shapelet portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of shapelets per `series_transformer` output.
        If None, extracts `int(np.sqrt(Xt.shape[2]) * 200 + 5)` shapelets where `Xt` is
        the series representation data.
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals and
        shapelets. If None, use the series as is. If "default", use [None, 1st Order
        Differences, PeriodogramTransformer, and ARCoefficientTransformer].

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    use_pycatch22 : bool, optional, default=False
        Wraps the C based pycatch22 implementation for aeon.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
    estimator : sklearn estimator, default=None
        An sklearn estimator to be built using the transformed data. Defaults to an
        extra trees forest with 200 trees.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    """

    @abstractmethod
    def __init__(
        self,
        n_intervals=None,
        n_shapelets=None,
        series_transformers="default",
        use_pycatch22=False,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_intervals = n_intervals
        self.n_shapelets = n_shapelets
        self.series_transformers = series_transformers
        self.use_pycatch22 = use_pycatch22
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, X, y) -> object:
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape

        rng = check_random_state(self.random_state)

        self._estimator = self.estimator
        if self.estimator is None:
            if is_classifier(self):
                self._estimator = ExtraTreesClassifier(
                    n_estimators=200, criterion="entropy"
                )
            elif is_regressor(self):
                self._estimator = ExtraTreesRegressor(n_estimators=200)
        # base_estimator must be an sklearn estimator
        elif not isinstance(self.estimator, BaseEstimator):
            raise ValueError(
                "base_estimator must be a scikit-learn BaseEstimator or None. "
                f"Found: {self.estimator}"
            )

        self._estimator = _clone_estimator(self._estimator, rng)

        n_jobs = check_n_jobs(self.n_jobs)
        m = getattr(self._estimator, "n_jobs", "missing")
        if m != "missing":
            self._estimator.n_jobs = n_jobs

        if self.series_transformers == "default":
            self._series_transformers = [
                None,
                FunctionTransformer(func=first_order_differences_3d, validate=False),
                PeriodogramTransformer(),
                ARCoefficientTransformer(
                    replace_nan=True, order=int(12 * (X.shape[2] / 100.0) ** 0.25)
                ),
            ]
        elif isinstance(self.series_transformers, (list, tuple)):
            self._series_transformers = [
                None if st is None else _clone_estimator(st, random_state=rng)
                for st in self.series_transformers
            ]
        else:
            self._series_transformers = [
                (
                    None
                    if self.series_transformers is None
                    else _clone_estimator(self.series_transformers, random_state=rng)
                )
            ]

        Xt = np.empty((X.shape[0], 0))
        self._transformers = []
        for st in self._series_transformers:
            if st is not None:
                m = getattr(st, "n_jobs", "missing")
                if m != "missing":
                    st.n_jobs = n_jobs

                s = st.fit_transform(X, y)
            else:
                s = X

            if self.n_intervals is None:
                n_intervals = int(np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5)
            elif callable(self.n_intervals):
                n_intervals = self.n_intervals(s)
            else:
                n_intervals = self.n_intervals

            ct = RandomIntervals(
                n_intervals=n_intervals,
                features=[
                    Catch22(
                        outlier_norm=True,
                        replace_nans=True,
                        use_pycatch22=self.use_pycatch22,
                    ),
                    row_mean,
                    row_std,
                    row_slope,
                    row_median,
                    row_iqr,
                    row_numba_min,
                    row_numba_max,
                    row_ppv,
                ],
                n_jobs=n_jobs,
            )
            _set_random_states(ct, rng)
            self._transformers.append(ct)
            t = ct.fit_transform(s, y)

            Xt = np.hstack((Xt, t))

            if self.n_shapelets is None:
                n_shapelets = int(np.sqrt(X.shape[2]) * 200 + 5)
            elif callable(self.n_shapelets):
                n_shapelets = self.n_shapelets(s)
            else:
                n_shapelets = self.n_shapelets

            st = RandomDilatedShapeletTransform(
                max_shapelets=n_shapelets, n_jobs=n_jobs
            )
            _set_random_states(st, rng)
            self._transformers.append(st)
            t = st.fit_transform(s, y)

            Xt = np.hstack((Xt, t))

        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

        self._estimator.fit(Xt, y)

        return self

    def _predict(self, X) -> np.ndarray:
        return self._estimator.predict(self._transform_data(X))

    def _predict_proba(self, X) -> np.ndarray:
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transform_data(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transform_data(X))
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists

    def _transform_data(self, X):
        Xt = np.empty((X.shape[0], 0))
        for i, st in enumerate(self._series_transformers):
            if st is not None:
                s = st.transform(X)
            else:
                s = X

            t = self._transformers[i * 2].transform(s)
            Xt = np.hstack((Xt, t))

            t = self._transformers[i * 2 + 1].transform(s)
            Xt = np.hstack((Xt, t))

        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        return Xt
