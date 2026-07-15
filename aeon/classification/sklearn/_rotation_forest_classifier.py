"""A Rotation Forest (RotF) vector classifier.

A Rotation Forest aeon implementation for continuous values only. Fits sklearn
conventions.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RotationForestClassifier"]

import numpy as np
from joblib import delayed
from sklearn.base import BaseEstimator, ClassifierMixin

from aeon.base._estimators.sklearn import BaseRotationForest
from aeon.utils._parallel import _run_jobs


class RotationForestClassifier(ClassifierMixin, BaseRotationForest):
    """
    A rotation forest (RotF) vector classifier.

    Implementation of the Rotation Forest classifier described [1]_. Builds a forest
    of trees build on random portions of the data transformed using PCA.

    Intended as a benchmark for time series data and a base classifier for
    transformation based approaches such as ShapeletTransformClassifier, this aeon
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
        `DecisionTreeClassifier` using entropy as a splitting measure. When set,
        the ``criterion``, ``splitter``, ``max_features``, ``max_depth``,
        ``max_leaf_nodes`` and ``min_samples_leaf`` parameters below are ignored.
    criterion : str, default="entropy"
        The ``criterion`` passed to the default ``DecisionTreeClassifier``.
        ``"gini"`` is faster to compute than ``"entropy"``. Only used when
        ``base_estimator`` is None.
    splitter : str, default="best"
        The ``splitter`` passed to the default ``DecisionTreeClassifier``.
        ``"random"`` is faster but less accurate. Only used when
        ``base_estimator`` is None.
    max_features : int, float, str or None, default=None
        The ``max_features`` passed to the default ``DecisionTreeClassifier``.
        ``None`` considers all (rotated) features at each split; a smaller value
        speeds up fitting. Only used when ``base_estimator`` is None.
    max_depth : int or None, default=None
        The ``max_depth`` passed to the default ``DecisionTreeClassifier``.
        Limiting depth speeds up fitting. Only used when ``base_estimator`` is
        None.
    max_leaf_nodes : int or None, default=None
        The ``max_leaf_nodes`` passed to the default ``DecisionTreeClassifier``.
        Only used when ``base_estimator`` is None.
    min_samples_leaf : int or float, default=1
        The ``min_samples_leaf`` passed to the default ``DecisionTreeClassifier``.
        Only used when ``base_estimator`` is None.
    pca_solver : str, default="deprecated"
        Has no effect. The group PCA is always computed with an exact
        eigendecomposition of the covariance matrix, equivalent to the
        scikit-learn PCA "full" solver.

        .. deprecated:: 1.6.0
            ``pca_solver`` has no effect and will be removed in v1.7.0.
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
    classes_ : list
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    n_cases_ : int
        The number of train cases in the training set.
    n_atts_ : int
        The number of attributes in the training set.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.

    See Also
    --------
    RotationForestRegressor

    References
    ----------
    .. [1] Rodriguez, Juan José, Ludmila I. Kuncheva, and Carlos J. Alonso. "Rotation
       forest: A new classifier ensemble method." IEEE transactions on pattern analysis
       and machine intelligence 28.10 (2006).

    .. [2] Bagnall, A., et al. "Is rotation forest the best classifier for problems
       with continuous features?." arXiv preprint arXiv:1809.06705 (2018).

    Examples
    --------
    >>> from aeon.classification.sklearn import RotationForestClassifier
    >>> from aeon.testing.data_generation import make_example_2d_numpy_collection
    >>> X, y = make_example_2d_numpy_collection(
    ...         n_cases=10, n_timepoints=12, random_state=0)
    >>> clf = RotationForestClassifier(n_estimators=10)
    >>> clf.fit(X, y)
    RotationForestClassifier(n_estimators=10)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    def __init__(
        self,
        n_estimators: int = 200,
        min_group: int = 3,
        max_group: int = 3,
        remove_proportion: float = 0.5,
        base_estimator: BaseEstimator | None = None,
        criterion: str = "entropy",
        splitter: str = "best",
        max_features=None,
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_leaf=1,
        pca_solver: str = "deprecated",
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
            criterion=criterion,
            splitter=splitter,
            max_features=max_features,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
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
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes ending in "_".
        """
        self._fit_rotf(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        return np.array(
            [self.classes_[int(np.argmax(prob))] for prob in self.predict_proba(X)]
        )

    def predict_proba(self, X) -> np.ndarray:
        """Probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        self._check_is_fitted()

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._prepare_predict_X(X)

        y_probas = _run_jobs(
            (
                delayed(self._predict_proba_for_estimator)(
                    X,
                    self.estimators_[i],
                    self._pcas[i],
                    self._groups[i],
                )
                for i in range(self._n_estimators)
            ),
            self._n_jobs,
            prefer="threads",
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes_) * self._n_estimators
        )
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
            The class labels.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes ending in "_".
        """
        return np.array(
            [
                self.classes_[int(np.argmax(prob))]
                for prob in self.fit_predict_proba(X, y)
            ]
        )

    def fit_predict_proba(self, X, y) -> np.ndarray:
        """Fit a forest of trees and estimate probabilities of the input.

        fit_predict produces prediction probability estimates using just the train
        data. The output is found using out-of-bag (OOB) estimates from the forest.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_cases, n_attributes]
            The training data.
        y : array-like, shape = [n_cases]
            The class labels.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes ending in "_".
        """
        return self._fit_predict_rotf(X, y)
