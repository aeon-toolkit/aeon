"""A Rotation Forest (RotF) vector regressor.

A Rotation Forest aeon implementation for continuous values only. Fits sklearn
conventions.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RotationForestRegressor"]

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin

from aeon.base._estimators.sklearn import BaseRotationForest


class RotationForestRegressor(RegressorMixin, BaseRotationForest):
    """
    A Rotation Forest (RotF) vector regressor.

    Implementation of the Rotation Forest regressor described in Rodriguez et al
    (2013) [1]. Builds a forest of trees build on random portions of the data
    transformed using PCA.

    Intended as a benchmark for time series data and a base regressor for
    transformation based approaches such as FreshPRINCERegressor, this aeon
    implementation only works with continuous attributes.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    min_group : int, default=3
        The minimum size of an attribute subsample group.
    max_group : int, default=3
        The maximum size of an attribute subsample group.
    remove_proportion : float, default=0.5
        The proportion of cases to be removed per group.
    base_estimator : BaseEstimator or None, default="None"
        Base estimator for the ensemble. By default, uses the sklearn
        `DecisionTreeRegressor` using MSE as a splitting measure.
    pca_solver : str, default="full"
        Deprecated and has no effect. The group PCA is computed with an exact
        eigendecomposition of the covariance matrix, equivalent to the
        scikit-learn PCA "full" solver. Will be removed in a future release.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding ``n_estimators``.
        Default of `0` means ``n_estimators`` is used.
    contract_max_n_estimators : int, default=500
        Max number of estimators to build when ``time_limit_in_minutes`` is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    n_cases_ : int
        The number of train cases in the training set.
    n_atts_ : int
        The number of attributes in the training set.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.

    See Also
    --------
    RotationForestClassifier

    Notes
    -----
    Predictions may differ slightly between scikit-learn versions. In particular,
    scikit-learn 1.8 fixed decision-tree handling of almost constant features, which
    can change the fitted trees used by ``RotationForestRegressor`` and therefore
    its output, even when using the same random state.

    References
    ----------
    .. [1] Rodriguez, Juan José, Ludmila I. Kuncheva, and Carlos J. Alonso. "Rotation
       forest: A new classifier ensemble method." IEEE transactions on pattern analysis
       and machine intelligence 28.10 (2006).

    .. [2] Bagnall, A., et al. "Is rotation forest the best classifier for problems
       with continuous features?." arXiv preprint arXiv:1809.06705 (2018).

    Examples
    --------
    >>> from aeon.regression.sklearn import RotationForestRegressor
    >>> from aeon.testing.data_generation import make_example_2d_numpy_collection
    >>> X, y = make_example_2d_numpy_collection(n_cases=10, n_timepoints=12,
    ...                              regression_target=True, random_state=0)
    >>> reg = RotationForestRegressor(n_estimators=10)
    >>> reg.fit(X, y)
    RotationForestRegressor(n_estimators=10)
    >>> reg.predict(X)
    array([0.7252543 , 1.50132442, 0.95608366, 1.64399016, 0.42385504,
           0.60639322, 1.01919317, 1.30157483, 1.66017354, 0.2900776 ])
    """

    def __init__(
        self,
        n_estimators: int = 200,
        min_group: int = 3,
        max_group: int = 3,
        remove_proportion: float = 0.5,
        base_estimator: BaseEstimator | None = None,
        pca_solver: str = "full",
        time_limit_in_minutes: float = 0.0,
        contract_max_n_estimators: int = 500,
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            base_estimator=base_estimator,
            pca_solver=pca_solver,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X, y):
        """Fit a forest of trees on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The training data.
        y : array-like, shape = [n_cases]
            The output values.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        self._fit_rotf(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict for all cases in X.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted output values.
        """
        self._check_is_fitted()

        X = self._prepare_predict_X(X)

        y_preds = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._predict_for_estimator)(
                X,
                self.estimators_[i],
                self._pcas[i],
                self._groups[i],
            )
            for i in range(self._n_estimators)
        )

        output = np.sum(y_preds, axis=0) / self._n_estimators

        return output

    def fit_predict(self, X, y) -> np.ndarray:
        """Fit a forest of trees and estimate predictions of the input.

        fit_predict produces prediction estimates using just the train data. The
        output is found using out-of-bag (OOB) estimates from the forest.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The training data.
        y : array-like, shape = [n_cases]
            The output values.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted output values.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        return self._fit_predict_rotf(X, y)
