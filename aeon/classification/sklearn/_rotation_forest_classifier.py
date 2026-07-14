"""Rotation Forest classifier for continuous tabular attributes."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RotationForestClassifier"]

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data

from aeon.base._base import _clone_estimator
from aeon.utils.validation import check_n_jobs


class RotationForestClassifier(ClassifierMixin, BaseEstimator):
    """Rotation Forest (RotF) vector classifier.

    Implementation of the Rotation Forest classifier described in [1]_. Each ensemble
    member partitions the attributes into random groups, fits PCA on a class-sensitive
    subsample for each group, and trains a classifier on the rotated attributes.

    This vector classifier is intended as a benchmark and as a base classifier for
    transformation-based approaches such as ``ShapeletTransformClassifier``. It accepts
    tabular data with continuous attributes rather than time-series collections.

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
        Estimator fitted to each rotated data set. If None, use
        ``DecisionTreeClassifier(criterion="entropy")``.
    pca_solver : str, default="auto"
        Solver to use for the PCA ``svd_solver`` parameter. See the scikit-learn PCA
        implementation for options.
    time_limit_in_minutes : float, default=0.0
        Time contract to limit build time in minutes, overriding ``n_estimators``.
        A value of 0 uses ``n_estimators``.
    contract_max_n_estimators : int, default=500
        Maximum number of estimators to build when ``time_limit_in_minutes`` is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    verbose : int, default=0
        Level of output printed during fit. Level 1 reports the fit configuration,
        periodic progress and a final summary. Level 2 and above additionally report
        every fitted estimator and estimated remaining time.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes_)
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    n_cases_ : int
        The number of train cases in the training set.
    n_atts_ : int
        The number of attributes in the training set.
    estimators_ : list of BaseEstimator
        The fitted estimators, with length ``n_estimators`` for a fixed-size fit.

    References
    ----------
    .. [1] Rodriguez, Juan Jose, Ludmila I. Kuncheva, and Carlos J. Alonso. "Rotation
       forest: A new classifier ensemble method." IEEE Transactions on Pattern Analysis
       and Machine Intelligence 28.10 (2006).

    .. [2] Bagnall, A., et al. "Is rotation forest the best classifier for problems
       with continuous features?" arXiv preprint arXiv:1809.06705 (2018).

    Examples
    --------
    >>> from aeon.classification.sklearn import RotationForestClassifier
    >>> from aeon.testing.data_generation import make_example_2d_numpy_collection
    >>> X, y = make_example_2d_numpy_collection(
    ...     n_cases=10, n_timepoints=12, random_state=0
    ... )
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
        pca_solver: str = "auto",
        time_limit_in_minutes: float = 0.0,
        contract_max_n_estimators: int = 500,
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
    ):
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.base_estimator = base_estimator
        self.pca_solver = pca_solver
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        super().__init__()

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
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        # data processing
        X = self._check_X(X)
        X = validate_data(self, X=X, reset=False, accept_sparse=False)

        # replace missing values with 0 and remove useless attributes
        X = X[:, self._useful_atts]

        # normalise the data.
        X = (X - self._min) / self._ptp

        y_probas = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.estimators_[i],
                self._pcas[i],
                self._groups[i],
            )
            for i in range(self._n_estimators)
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
        X_t = self._fit_rotf(X, y, save_transformed_data=True)

        rng = check_random_state(self.random_state)

        p = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._train_probas_for_estimator)(
                X_t,
                y,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(self._n_estimators)
        )
        y_probas, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        oob_arrays = [np.asarray(o, dtype=np.intp) for o in oobs if len(o) > 0]
        if oob_arrays:
            divisors = np.bincount(
                np.concatenate(oob_arrays), minlength=self.n_cases_
            ).astype(float)
        else:
            divisors = np.zeros(self.n_cases_, dtype=float)

        nonzero = divisors > 0
        results[nonzero] = results[nonzero] / divisors[nonzero, None]
        results[~nonzero] = 1 / self.n_classes_

        return results

    def _fit_rotf(self, X, y, save_transformed_data: bool = False):
        # data processing
        X = self._check_X(X)
        X, y = validate_data(self, X=X, y=y, ensure_min_samples=2, accept_sparse=False)
        check_classification_targets(y)

        self.n_cases_, self.n_atts_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        # escape if only one class seen
        if self.n_classes_ == 1:
            self._is_fitted = True
            return self

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        log_each_estimator = self.verbose >= 2
        log_progress = self.verbose == 1
        if self.verbose > 0:
            if time_limit > 0:
                fit_limit = (
                    f"time_limit={self._format_duration(time_limit)}, "
                    f"max_n_estimators={self.contract_max_n_estimators}"
                )
            else:
                fit_limit = f"n_estimators={self.n_estimators}"
            self._log(
                f"[RotF] Starting fit: n_cases={self.n_cases_}, "
                f"n_attributes={self.n_atts_}, {fit_limit}, n_jobs={self._n_jobs}"
            )

        self._base_estimator = self.base_estimator
        if self.base_estimator is None:
            self._base_estimator = DecisionTreeClassifier(criterion="entropy")

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

        X_cls_split = [X[np.where(y == i)] for i in self.classes_]

        rng = check_random_state(self.random_state)

        if time_limit > 0:
            if log_progress:
                progress_interval = time_limit / 10
                next_progress = progress_interval

            self._n_estimators = 0
            self.estimators_ = []
            self._pcas = []
            self._groups = []
            X_t = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
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

                if log_each_estimator:
                    contract_remaining = self._format_duration(
                        max(0.0, time_limit - train_time)
                    )
                    first_estimator = self._n_estimators - len(fit) + 1
                    for estimator_idx in range(first_estimator, self._n_estimators + 1):
                        self._log(
                            f"[RotF] Estimator {estimator_idx}: "
                            f"elapsed={train_time:.2f}s, "
                            f"contract_remaining={contract_remaining}"
                        )
                elif log_progress and train_time >= next_progress:
                    self._log(
                        f"[RotF] Progress: built={self._n_estimators}, "
                        f"elapsed={train_time:.2f}s"
                    )
                    next_progress = train_time + progress_interval
        else:
            self._n_estimators = self.n_estimators
            if self.verbose > 0:
                estimator_start_time = time.time()
                if log_each_estimator:
                    batch_size = self._n_jobs
                else:
                    batch_size = max(self._n_jobs, (self._n_estimators + 9) // 10)

                fit = []
                for batch_start in range(0, self._n_estimators, batch_size):
                    current_batch_size = min(
                        batch_size, self._n_estimators - batch_start
                    )
                    batch_fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                        delayed(self._fit_estimator)(
                            X,
                            X_cls_split,
                            y,
                            check_random_state(rng.randint(np.iinfo(np.int32).max)),
                            save_transformed_data,
                        )
                        for _ in range(current_batch_size)
                    )
                    fit.extend(batch_fit)

                    built = len(fit)
                    estimator_elapsed = time.time() - estimator_start_time
                    if log_each_estimator:
                        if built == 1:
                            time_estimate = "estimated_remaining=estimating"
                        else:
                            estimated_remaining = (estimator_elapsed / built) * (
                                self._n_estimators - built
                            )
                            time_estimate = (
                                "estimated_remaining="
                                f"{self._format_duration(estimated_remaining)}"
                            )
                        elapsed = time.time() - start_time
                        for estimator_idx in range(
                            batch_start + 1, batch_start + current_batch_size + 1
                        ):
                            self._log(
                                f"[RotF] Estimator "
                                f"{estimator_idx}/{self._n_estimators}: "
                                f"elapsed={elapsed:.2f}s, {time_estimate}"
                            )
                    else:
                        self._log(
                            f"[RotF] Progress: built={built}/{self._n_estimators}, "
                            f"elapsed={time.time() - start_time:.2f}s"
                        )
            else:
                fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
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
        if self.verbose > 0:
            self._log(
                f"[RotF] Finished fit: built={len(self.estimators_)}, "
                f"elapsed={time.time() - start_time:.2f}s"
            )
        return X_t

    @staticmethod
    def _log(message):
        """Print a fit progress message after the caller checks verbosity."""
        print(message, flush=True)  # noqa: T201

    @staticmethod
    def _format_duration(seconds):
        """Format a duration for concise progress output."""
        if seconds < 10:
            return f"{seconds:.2f}s"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes, remaining_seconds = divmod(seconds, 60)
            return f"{int(minutes)}m {remaining_seconds:.0f}s"

        hours, remaining_seconds = divmod(seconds, 3600)
        minutes = remaining_seconds // 60
        return f"{int(hours)}h {int(minutes)}m"

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
            classes = rng.choice(
                range(self.n_classes_),
                size=rng.randint(1, self.n_classes_ + 1),
                replace=False,
            )

            # randomly add the classes with the randomly selected attributes.
            X_t = np.zeros((0, len(group)))
            for cls_idx in classes:
                c = X_cls_split[cls_idx]
                X_t = np.concatenate((X_t, c[:, group]), axis=0)

            sample_ind = rng.choice(
                X_t.shape[0],
                max(1, int(X_t.shape[0] * self.remove_proportion)),
                replace=False,
            )
            X_t = X_t[sample_ind]

            # try to fit the PCA if it fails, remake it, and add 10 random data
            # instances.
            while True:
                # ignore err state on PCA because we account if it fails.
                with np.errstate(divide="ignore", invalid="ignore"):
                    # differences between os occasionally. seems to happen when there
                    # are low amounts of cases in the fit
                    pca = PCA(random_state=rng, svd_solver=self.pca_solver).fit(X_t)

                if not np.isnan(pca.explained_variance_ratio_).all():
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)

        # merge all the pca_transformed data into one instance and build a classifier
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

        return tree, pcas, groups, X_t if save_transformed_data else None

    def _predict_proba_for_estimator(self, X, clf: int, pcas: PCA, groups):
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        X_t = X_t.astype(np.float32)
        X_t = np.nan_to_num(
            X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
        )

        probas = clf.predict_proba(X_t)

        if probas.shape[1] != self.n_classes_:
            new_probas = np.zeros((probas.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas

        return probas

    def _train_probas_for_estimator(self, X_t, y, idx, rng: np.random.RandomState):
        indices = range(self.n_cases_)
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = [n for n in indices if n not in subsample]

        results = np.zeros((self.n_cases_, self.n_classes_))
        if len(oob) == 0:
            return [results, oob]

        clf = _clone_estimator(self._base_estimator, rng)
        clf.fit(X_t[idx][subsample], y[subsample])
        probas = clf.predict_proba(X_t[idx][oob])

        if probas.shape[1] != self.n_classes_:
            new_probas = np.zeros((probas.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas

        for n, proba in enumerate(probas):
            results[oob[n]] += proba

        return [results, oob]

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

        msg = (
            "RotationForestClassifier is not a time series classifier. "
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
