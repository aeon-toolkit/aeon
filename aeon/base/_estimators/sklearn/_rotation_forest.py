"""A base class for Rotation Forest (RotF) estimators.

A Rotation Forest aeon implementation for continuous values only. Fits sklearn
conventions. Contains the code shared by the classifier and regressor.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseRotationForest"]

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data

from aeon.base._base import _clone_estimator
from aeon.utils.validation import check_n_jobs


class _GroupPCA:
    """Minimal PCA via eigendecomposition of the covariance matrix.

    A lightweight replacement for the sklearn PCA previously used to rotate each
    attribute group. Groups are only a few attributes wide, so an exact
    eigendecomposition of the small covariance matrix is much faster than a full
    SVD of the data matrix and avoids all sklearn fit/transform overhead.

    Matches the components (and sign convention) of the sklearn PCA "full"
    solver, except for rank-deficient inputs where the trailing components span
    the same null space but may differ in direction.

    ``components_`` is ``None`` if the input is degenerate (fewer than two cases
    or zero total variance), mirroring the all-NaN ``explained_variance_ratio_``
    failure previously used to trigger a refit.
    """

    __slots__ = ("mean_", "components_")

    def fit(self, X):
        n_cases, n_atts = X.shape
        self.mean_ = X.mean(axis=0)
        self.components_ = None

        if n_cases < 2:
            return self

        X_c = X - self.mean_
        cov = (X_c.T @ X_c) / (n_cases - 1)
        total_var = np.trace(cov)
        if not np.isfinite(total_var) or total_var <= 0:
            return self

        _, vecs = np.linalg.eigh(cov)
        # rows = components in descending eigenvalue order, truncated to the
        # number of components the sklearn PCA default would keep
        vecs = vecs[:, ::-1].T[: min(n_cases, n_atts)]

        # sklearn svd_flip sign convention: the largest absolute value in each
        # component is positive
        max_abs = np.argmax(np.abs(vecs), axis=1)
        signs = np.sign(vecs[np.arange(vecs.shape[0]), max_abs])
        vecs *= signs[:, None]

        self.components_ = vecs
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


class BaseRotationForest(BaseEstimator):
    """A base class for Rotation Forest (RotF) estimators.

    Implements the code shared by ``RotationForestClassifier`` and
    ``RotationForestRegressor`` [1]_. Builds a forest of trees build on random
    portions of the data transformed using PCA. Classification and regression
    specific behaviour is switched on the subclass using the sklearn
    ``is_classifier`` function.

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
    base_estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble. By default, uses the sklearn
        ``DecisionTreeClassifier`` using entropy as a splitting measure for
        classifiers and the sklearn ``DecisionTreeRegressor`` using MSE as a
        splitting measure for regressors. When set, the ``criterion``,
        ``splitter``, ``max_features``, ``max_depth``, ``max_leaf_nodes`` and
        ``min_samples_leaf`` parameters below are ignored.
    criterion : str
        The ``criterion`` passed to the default decision tree. Defaults to
        ``"entropy"`` for classifiers and ``"squared_error"`` for regressors.
        Only used when ``base_estimator`` is None.
    splitter : str, default="best"
        The ``splitter`` passed to the default decision tree. ``"random"`` is
        faster but less accurate. Only used when ``base_estimator`` is None.
    max_features : int, float, str or None, default=None
        The ``max_features`` passed to the default decision tree. ``None`` uses
        all (rotated) features at each split; a smaller value speeds up fitting.
        Only used when ``base_estimator`` is None.
    max_depth : int or None, default=None
        The ``max_depth`` passed to the default decision tree. Limiting depth
        speeds up fitting. Only used when ``base_estimator`` is None.
    max_leaf_nodes : int or None, default=None
        The ``max_leaf_nodes`` passed to the default decision tree. Only used
        when ``base_estimator`` is None.
    min_samples_leaf : int or float, default=1
        The ``min_samples_leaf`` passed to the default decision tree. Only used
        when ``base_estimator`` is None.
    pca_solver : str, default="auto"
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

    References
    ----------
    .. [1] Rodriguez, Juan José, Ludmila I. Kuncheva, and Carlos J. Alonso. "Rotation
       forest: A new classifier ensemble method." IEEE transactions on pattern analysis
       and machine intelligence 28.10 (2006).
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
        pca_solver: str = "auto",
        time_limit_in_minutes: float = 0.0,
        contract_max_n_estimators: int = 500,
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.pca_solver = pca_solver
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _parallel(self, jobs):
        """Run a sequence of ``delayed`` jobs, skipping joblib when single-threaded.

        ``jobs`` is an iterable of ``delayed(func)(*args, **kwargs)`` tuples. When
        ``n_jobs == 1`` these are called directly, avoiding joblib's per-task
        dispatch overhead while preserving the order in which the jobs (and hence
        any random draws in their arguments) are produced.
        """
        if self._n_jobs == 1:
            return [func(*args, **kwargs) for func, args, kwargs in jobs]
        return Parallel(n_jobs=self._n_jobs, prefer="threads")(jobs)

    def _fit_rotf(self, X, y, save_transformed_data: bool = False):
        # cache the task type once rather than calling is_classifier repeatedly
        # in the per-group/per-estimator inner loops
        self._is_classifier = is_classifier(self)

        # data processing
        X = self._check_X(X)
        X, y = validate_data(self, X=X, y=y, ensure_min_samples=2, accept_sparse=False)

        if self._is_classifier:
            check_classification_targets(y)
        else:
            self._label_average = np.mean(y)

        self.n_cases_, self.n_atts_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        if self._is_classifier:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            self._class_dictionary = {}
            for index, class_val in enumerate(self.classes_):
                self._class_dictionary[class_val] = index

            # escape if only one class seen
            if self.n_classes_ == 1:
                self._is_fitted = True
                return self

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        self._base_estimator = self.base_estimator
        if self.base_estimator is None:
            self._base_estimator = self._new_default_tree()

        # sklearn decision trees can skip input validation, as we always feed
        # them a finite, contiguous float32 array we built ourselves
        self._skip_tree_checks = isinstance(self._base_estimator, BaseDecisionTree)

        # remove useless attributes
        self._useful_atts = ~np.all(X[1:] == X[:-1], axis=0)
        X = X[:, self._useful_atts]
        if sum(self._useful_atts) == 0:
            raise ValueError(
                "All attributes in X contain the same value.",
            )

        self._n_atts = X.shape[1]

        # normalise attributes
        self._min = X.min(axis=0)
        self._ptp = X.max(axis=0) - self._min
        X = (X - self._min) / self._ptp

        X_cls_split = (
            [X[np.where(y == i)] for i in self.classes_]
            if self._is_classifier
            else None
        )

        rng = check_random_state(self.random_state)

        if time_limit > 0:
            self._n_estimators = 0
            self.estimators_ = []
            self._pcas = []
            self._groups = []
            X_t = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = self._parallel(
                    delayed(self._fit_estimator)(
                        X,
                        X_cls_split,
                        y,
                        check_random_state(rng.randint(np.iinfo(np.int32).max)),
                        save_transformed_data,
                    )
                    for _ in range(self._n_jobs)
                )

                estimators, pcas, groups, transformed_data = zip(*fit)

                self.estimators_ += estimators
                self._pcas += pcas
                self._groups += groups
                X_t += transformed_data

                self._n_estimators += self._n_jobs
                train_time = time.time() - start_time
        else:
            self._n_estimators = self.n_estimators

            fit = self._parallel(
                delayed(self._fit_estimator)(
                    X,
                    X_cls_split,
                    y,
                    check_random_state(rng.randint(np.iinfo(np.int32).max)),
                    save_transformed_data,
                )
                for _ in range(self._n_estimators)
            )

            self.estimators_, self._pcas, self._groups, X_t = zip(*fit)

        self._is_fitted = True
        return X_t

    def _fit_estimator(
        self,
        X,
        X_cls_split,
        y,
        rng: np.random.RandomState,
        save_transformed_data: bool,
    ):
        groups = self._generate_groups(rng)
        pcas = []

        # construct the slices to fit the PCAs too.
        for group in groups:
            X_t = self._sample_group_cases(X, X_cls_split, group, rng)

            # try to fit the PCA. if the data is degenerate, remake it, adding
            # 10 random data instances until it fits.
            while True:
                pca = _GroupPCA().fit(X_t)

                if pca.components_ is not None:
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)

        # merge all the pca_transformed data into one instance and build an estimator
        # on it.
        X_t = self._transform_for_estimator(X, pcas, groups)

        tree = self._make_tree(rng)
        if self._skip_tree_checks:
            # X_t is already a validated, finite, contiguous float32 array
            tree.fit(X_t, y, check_input=False)
        else:
            tree.fit(X_t, y)

        return tree, pcas, groups, X_t if save_transformed_data else None

    def _new_default_tree(self, random_state=None):
        """Build the default decision tree from the exposed tree parameters."""
        tree_cls = (
            DecisionTreeClassifier if self._is_classifier else DecisionTreeRegressor
        )
        return tree_cls(
            criterion=self.criterion,
            splitter=self.splitter,
            max_features=self.max_features,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state,
        )

    def _make_tree(self, rng: np.random.RandomState):
        """Construct a base estimator, avoiding clone overhead for the default.

        The default tree is seeded with an int drawn from ``rng``, consuming
        one draw exactly as the ``_clone_estimator`` path does via the sklearn
        ``_set_random_states``, keeping the two routes interchangeable.
        """
        if self.base_estimator is None:
            return self._new_default_tree(
                random_state=rng.randint(np.iinfo(np.int32).max)
            )
        return _clone_estimator(self._base_estimator, random_state=rng)

    def _sample_group_cases(self, X, X_cls_split, group, rng: np.random.RandomState):
        """Select the subsample of cases used to fit the PCA for a single group."""
        if self._is_classifier:
            classes = rng.choice(
                range(self.n_classes_),
                size=rng.randint(1, self.n_classes_ + 1),
                replace=False,
            )

            # gather the selected classes' attributes with a single concatenate
            # rather than growing the array one class at a time
            X_t = np.concatenate(
                [X_cls_split[cls_idx][:, group] for cls_idx in classes], axis=0
            )

            sample_ind = rng.choice(
                X_t.shape[0],
                max(1, int(X_t.shape[0] * self.remove_proportion)),
                replace=False,
            )
            return X_t[sample_ind]
        else:
            sample_ind = rng.choice(
                X.shape[0],
                max(1, int(X.shape[0] * self.remove_proportion)),
                replace=False,
            )

            X_t = X[sample_ind, :]
            return X_t[:, group]

    def _transform_for_estimator(self, X, pcas, groups):
        """Apply the rotation for a single ensemble member to X.

        Writes each group's rotation straight into a preallocated float32 array
        rather than building and concatenating a list of per-group arrays. The
        float64 -> float32 cast happens on assignment, matching a trailing
        ``astype``.
        """
        n_cases = X.shape[0]
        n_comps = sum(pca.components_.shape[0] for pca in pcas)
        X_t = np.empty((n_cases, n_comps), dtype=np.float32)
        pos = 0
        for pca, group in zip(pcas, groups):
            rot = pca.transform(X[:, group])
            width = rot.shape[1]
            X_t[:, pos : pos + width] = rot
            pos += width

        # X_t can only be non-finite if the cast to float32 overflowed, which
        # cannot happen for the normalised inputs and finite components here.
        # Run the full replacement only in that (essentially unreachable) case
        # so the common path pays a single finiteness scan instead of three.
        if not np.isfinite(X_t).all():
            np.nan_to_num(
                X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
            )
        return X_t

    def _predict_proba_for_estimator(self, X, clf, pcas, groups):
        X_t = self._transform_for_estimator(X, pcas, groups)

        if self._skip_tree_checks:
            probas = clf.predict_proba(X_t, check_input=False)
        else:
            probas = clf.predict_proba(X_t)

        if probas.shape[1] != self.n_classes_:
            new_probas = np.zeros((probas.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas

        return probas

    def _predict_for_estimator(self, X, clf, pcas, groups):
        X_t = self._transform_for_estimator(X, pcas, groups)
        if self._skip_tree_checks:
            return clf.predict(X_t, check_input=False)
        return clf.predict(X_t)

    def _train_probas_for_estimator(self, X_t, y, idx, rng: np.random.RandomState):
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        in_bag = np.zeros(self.n_cases_, dtype=bool)
        in_bag[subsample] = True
        oob = np.flatnonzero(~in_bag)

        results = np.zeros((self.n_cases_, self.n_classes_))
        if len(oob) == 0:
            return [results, oob]

        clf = self._make_tree(rng)
        if self._skip_tree_checks:
            clf.fit(X_t[idx][subsample], y[subsample], check_input=False)
            probas = clf.predict_proba(X_t[idx][oob], check_input=False)
        else:
            clf.fit(X_t[idx][subsample], y[subsample])
            probas = clf.predict_proba(X_t[idx][oob])

        if probas.shape[1] != self.n_classes_:
            new_probas = np.zeros((probas.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas

        results[oob] = probas

        return [results, oob]

    def _train_preds_for_estimator(self, X_t, y, idx, rng: np.random.RandomState):
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        in_bag = np.zeros(self.n_cases_, dtype=bool)
        in_bag[subsample] = True
        oob = np.flatnonzero(~in_bag)

        results = np.zeros(self.n_cases_)
        if len(oob) == 0:
            return [results, oob]

        clf = self._make_tree(rng)
        if self._skip_tree_checks:
            clf.fit(X_t[idx][subsample], y[subsample], check_input=False)
            preds = clf.predict(X_t[idx][oob], check_input=False)
        else:
            clf.fit(X_t[idx][subsample], y[subsample])
            preds = clf.predict(X_t[idx][oob])

        results[oob] = preds

        return [results, oob]

    def _fit_predict_rotf(self, X, y):
        """Fit the forest and estimate train predictions using OOB estimates.

        Returns class probabilities for classifiers and predicted values for
        regressors.
        """
        X_t = self._fit_rotf(X, y, save_transformed_data=True)

        rng = check_random_state(self.random_state)

        train_fn = (
            self._train_probas_for_estimator
            if self._is_classifier
            else self._train_preds_for_estimator
        )
        p = self._parallel(
            delayed(train_fn)(
                X_t,
                y,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(self._n_estimators)
        )
        y_preds, oobs = zip(*p)

        results = np.sum(y_preds, axis=0)
        oob_arrays = [np.asarray(o, dtype=np.intp) for o in oobs if len(o) > 0]
        if oob_arrays:
            divisors = np.bincount(
                np.concatenate(oob_arrays), minlength=self.n_cases_
            ).astype(float)
        else:
            divisors = np.zeros(self.n_cases_, dtype=float)

        nonzero = divisors > 0
        if is_classifier(self):
            results[nonzero] = results[nonzero] / divisors[nonzero, None]
            results[~nonzero] = 1 / self.n_classes_
        else:
            results[nonzero] = results[nonzero] / divisors[nonzero]
            results[~nonzero] = self._label_average

        return results

    def _check_is_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

    def _prepare_predict_X(self, X):
        """Validate, subset and normalise X for prediction."""
        X = self._check_X(X)
        X = validate_data(self, X=X, reset=False, accept_sparse=False)

        # replace missing values with 0 and remove useless attributes
        X = X[:, self._useful_atts]

        # normalise the data.
        return (X - self._min) / self._ptp

    def _generate_groups(self, rng: np.random.RandomState):
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

    def _check_X(self, X):
        if issparse(X):
            return X

        task = "classifier" if is_classifier(self) else "regressor"
        msg = (
            f"{self.__class__.__name__} is not a time series {task}. "
            "A valid sklearn input such as a 2d numpy array is required."
            "Sparse input formats are currently not supported."
        )
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        else:
            try:
                X = np.array(X)
            except Exception:
                raise ValueError(msg)

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(msg)

        return X
