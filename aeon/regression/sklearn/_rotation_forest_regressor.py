"""A Rotation Forest (RotF) vector regressor.

A Rotation Forest aeon implementation for continuous values only. Fits sklearn
conventions.
"""

__maintainer__ = []
__all__ = ["RotationForestRegressor"]

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.exceptions import NotFittedError
from aeon.utils.validation import check_n_jobs


class RotationForestRegressor(BaseEstimator):
    """
    A Rotation Forest (RotF) vector regressor.

    Implementation of the Rotation Forest regressor described in Rodriguez et al
    (2013) [1]. Builds a forest of trees build on random portions of the data
    transformed using PCA.

    Intended as a benchmark for time series data and a base regressor for
    transformation based appraoches such as FreshPRINCERegressor, this aeon
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
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding ``n_estimators``.
        Default of `0` means ``n_estimators`` is used.
    contract_max_n_estimators : int, default=500
        Max number of estimators to build when ``time_limit_in_minutes`` is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit in ``transformed_data_`` for use in
        ``_get_train_probs``.
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
    transformed_data_ : list of shape (n_estimators) of ndarray
        The transformed training dataset for all regressors. Only saved when
        ``save_transformed_data`` is `True`.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.

    See Also
    --------
    FreshPRINCERegressor: A feature-based regressor using Rotation Forest.

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
    >>> from aeon.testing.utils.data_gen import make_example_2d_numpy
    >>> X, y = make_example_2d_numpy(n_cases=10, n_timepoints=12,
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
        n_estimators=200,
        min_group=3,
        max_group=3,
        remove_proportion=0.5,
        base_estimator=None,
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.base_estimator = base_estimator
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

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
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForestRegressor is not a time series regressor. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)

        self._label_average = np.mean(y)

        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_cases_, self.n_atts_ = X.shape

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        if self.base_estimator is None:
            self._base_estimator = DecisionTreeRegressor(criterion="squared_error")

        # remove useless attributes
        self._useful_atts = ~np.all(X[1:] == X[:-1], axis=0)
        X = X[:, self._useful_atts]

        self._n_atts = X.shape[1]

        # normalise attributes
        self._min = X.min(axis=0)
        self._ptp = X.max(axis=0) - self._min
        X = (X - self._min) / self._ptp

        rng = check_random_state(self.random_state)

        if time_limit > 0:
            self._n_estimators = 0
            self.estimators_ = []
            self._pcas = []
            self._groups = []
            self.transformed_data_ = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                    delayed(self._fit_estimator)(
                        X,
                        y,
                        check_random_state(rng.randint(np.iinfo(np.int32).max)),
                    )
                    for _ in range(self._n_jobs)
                )

                estimators, pcas, groups, transformed_data = zip(*fit)

                self.estimators_ += estimators
                self._pcas += pcas
                self._groups += groups
                self.transformed_data_ += transformed_data

                self._n_estimators += self._n_jobs
                train_time = time.time() - start_time
        else:
            self._n_estimators = self.n_estimators

            fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(self._fit_estimator)(
                    X,
                    y,
                    check_random_state(rng.randint(np.iinfo(np.int32).max)),
                )
                for _ in range(self._n_estimators)
            )

            self.estimators_, self._pcas, self._groups, self.transformed_data_ = zip(
                *fit
            )

        self._is_fitted = True
        return self

    def predict(self, X):
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
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForestRegressor is not a time series regressor. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        # replace missing values with 0 and remove useless attributes
        X = X[:, self._useful_atts]

        # normalise the data.
        X = (X - self._min) / self._ptp

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

    def _get_train_preds(self, X, y):
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForestRegressor is not a time series regressor. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        n_cases, n_atts = X.shape

        if n_cases != self.n_cases_ or n_atts != self.n_atts_:
            raise ValueError(
                "n_cases, n_atts mismatch. X should be the same as the training "
                "data used in fit for generating train predictions."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        rng = check_random_state(self.random_state)

        p = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._train_preds_for_estimator)(
                y,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(self._n_estimators)
        )
        y_preds, oobs = zip(*p)

        results = np.sum(y_preds, axis=0)
        divisors = np.zeros(n_cases)
        for oob in oobs:
            for inst in oob:
                divisors[inst] += 1

        for i in range(n_cases):
            results[i] = (
                self._label_average if divisors[i] == 0 else results[i] / divisors[i]
            )

        return results

    def _fit_estimator(self, X, y, rng):
        groups = self._generate_groups(rng)
        pcas = []

        # construct the slices to fit the PCAs too.
        for group in groups:
            sample_ind = rng.choice(
                X.shape[0],
                max(1, int(X.shape[0] * self.remove_proportion)),
                replace=False,
            )

            X_t = X[sample_ind, :]
            X_t = X_t[:, group]

            # try to fit the PCA if it fails, remake it, and add 10 random data
            # instances.
            while True:
                # ignore err state on PCA because we account if it fails.
                with np.errstate(divide="ignore", invalid="ignore"):
                    # differences between os occasionally. seems to happen when there
                    # are low amounts of cases in the fit
                    pca = PCA(random_state=rng).fit(X_t)

                if not np.isnan(pca.explained_variance_ratio_).all():
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)

        # merge all the pca_transformed data into one instance and build a regressor
        # on it.
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        X_t = X_t.astype(np.float32)
        X_t = np.nan_to_num(
            X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
        )

        tree = _clone_estimator(self._base_estimator, random_state=rng)
        tree.fit(X_t, y)

        return tree, pcas, groups, X_t if self.save_transformed_data else None

    def _predict_for_estimator(self, X, clf, pcas, groups):
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        X_t = X_t.astype(np.float32)
        X_t = np.nan_to_num(
            X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
        )

        return clf.predict(X_t)

    def _train_preds_for_estimator(self, y, idx, rng):
        indices = range(self.n_cases_)
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = [n for n in indices if n not in subsample]

        results = np.zeros(self.n_cases_)
        if len(oob) == 0:
            return [results, oob]

        clf = _clone_estimator(self._base_estimator, rng)
        clf.fit(self.transformed_data_[idx][subsample], y[subsample])
        preds = clf.predict(self.transformed_data_[idx][oob])

        for n, pred in enumerate(preds):
            results[oob[n]] += pred

        return [results, oob]

    def _generate_groups(self, rng):
        permutation = rng.permutation(np.arange(0, self._n_atts))

        # select the size of each group.
        group_size_count = np.zeros(self.max_group - self.min_group + 1)
        n_attributes = 0
        n_groups = 0
        while n_attributes < self._n_atts:
            n = rng.randint(group_size_count.shape[0])
            group_size_count[n] += 1
            n_attributes += self.min_group + n
            n_groups += 1

        groups = []
        current_attribute = 0
        current_size = 0
        for i in range(0, n_groups):
            while group_size_count[current_size] == 0:
                current_size += 1
            group_size_count[current_size] -= 1

            n = self.min_group + current_size
            groups.append(np.zeros(n, dtype=int))
            for k in range(0, n):
                if current_attribute < permutation.shape[0]:
                    groups[i][k] = permutation[current_attribute]
                else:
                    groups[i][k] = permutation[rng.randint(permutation.shape[0])]
                current_attribute += 1

        return groups
