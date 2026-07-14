"""Arsenal classifier.

Kernel-based ensemble of ROCKET classifiers.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["Arsenal"]

import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MultiRocket,
    Rocket,
)
from aeon.utils.validation import check_n_jobs


class Arsenal(BaseClassifier):
    """Arsenal ensemble.

    Arsenal fits an ensemble of ROCKET transforms followed by ``RidgeClassifierCV``
    classifiers. It weights each classifier by its internal cross-validation accuracy,
    enabling probability estimates at the cost of lower scalability than a single
    ``RocketClassifier`` [1]_.

    Parameters
    ----------
    n_kernels : int, default=2000
        Number of kernels for each ROCKET transform.
    n_estimators : int, default=25
        Number of estimators to build for the ensemble.
    rocket_transform : str, default="rocket"
        ROCKET transform used by each ensemble member. Valid values are ``"rocket"``,
        ``"minirocket"``, and ``"multirocket"``.
    max_dilations_per_kernel : int, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=4
        MultiRocket only. The number of features per kernel.
    time_limit_in_minutes : float, default=0.0
        Time contract for fitting, in minutes, overriding ``n_estimators``. A value of 0
        uses ``n_estimators``.
    contract_max_n_estimators : int, default=100
        Maximum number of estimators when ``time_limit_in_minutes`` is set.
    class_weight : dict, "balanced" or None, default=None
        Class weights passed to each ``RidgeClassifierCV``. If None, all classes have
        weight one. ``"balanced"`` weights classes inversely to their frequencies in
        the training data.
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
    n_classes_ : int
        The number of classes.
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The length of each series.
    classes_ : np.ndarray of shape (n_classes_)
        The class labels.
    estimators_ : list of Pipeline
        The fitted ROCKET, scaler, and ridge-classifier pipelines.
    weights_ : list of float
        Cross-validation accuracy of each fitted pipeline.
    n_estimators_ : int
        The number of estimators in the ensemble.

    See Also
    --------
    RocketClassifier
        Arsenal is an ensemble of RocketClassifier.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/kernel_based/Arsenal.java>`_.

    References
    ----------
    .. [1] Middlehurst, M., Large, J., Flynn, M. et al.
       "HIVE-COTE 2.0: a new meta ensemble for time series classification."
       Machine Learning 110, 3211--3243 (2021).
       https://doi.org/10.1007/s10994-021-06057-9

    Examples
    --------
    >>> from aeon.classification.convolution_based import Arsenal
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = Arsenal(n_kernels=100, n_estimators=5)
    >>> clf.fit(X_train, y_train)
    Arsenal(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels: int = 2000,
        n_estimators: int = 25,
        rocket_transform: str = "rocket",
        max_dilations_per_kernel: int = 32,
        n_features_per_kernel: int = 4,
        time_limit_in_minutes: float = 0.0,
        contract_max_n_estimators: int = 100,
        class_weight=None,
        n_jobs: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_kernels = n_kernels
        self.n_estimators = n_estimators
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators

        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0
        self.estimators_ = []
        self.weights_ = []

        self._weight_sum = 0

        super().__init__()

    def _fit(self, X, y):
        """Fit Arsenal to training data.

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
        self._fit_arsenal(X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        y_probas = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.estimators_[i],
                i,
            )
            for i in range(self.n_estimators_)
        )

        return np.around(
            np.sum(y_probas, axis=0) / (np.ones(self.n_classes_) * self._weight_sum), 8
        )

    def _fit_predict(self, X, y) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._fit_predict_proba(X, y)
            ]
        )

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        Xt = self._fit_arsenal(X, y, keep_transformed_data=True)

        rng = check_random_state(self.random_state)

        p = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._train_probas_for_estimator)(
                Xt,
                y,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(self.n_estimators_)
        )
        y_probas, weights, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        divisors = np.zeros(self.n_cases_)
        for n, oob in enumerate(oobs):
            for inst in oob:
                divisors[inst] += weights[n]

        for i in range(self.n_cases_):
            results[i] = (
                np.ones(self.n_classes_) * (1 / self.n_classes_)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes_) * divisors[i])
            )

        return results

    def _fit_arsenal(self, X, y, keep_transformed_data=False):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

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
                f"[Arsenal] Starting fit: n_cases={self.n_cases_}, "
                f"n_channels={self.n_channels_}, "
                f"n_timepoints={self.n_timepoints_}, "
                f"transform={self.rocket_transform}, n_kernels={self.n_kernels}, "
                f"{fit_limit}, n_jobs={self._n_jobs}"
            )

        if self.rocket_transform == "rocket":
            base_rocket = Rocket(n_kernels=self.n_kernels)
        elif self.rocket_transform == "minirocket":
            base_rocket = MiniRocket(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
            )
        elif self.rocket_transform == "multirocket":
            base_rocket = MultiRocket(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
                n_features_per_kernel=self.n_features_per_kernel,
            )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")

        rng = check_random_state(self.random_state)

        if time_limit > 0:
            if log_progress:
                progress_interval = time_limit / 10
                next_progress = progress_interval

            self.n_estimators_ = 0
            self.estimators_ = []
            Xt = []

            while (
                train_time < time_limit
                and self.n_estimators_ < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                    delayed(self._fit_ensemble_estimator)(
                        _clone_estimator(
                            base_rocket, rng.randint(np.iinfo(np.int32).max)
                        ),
                        X,
                        y,
                        keep_transformed_data=keep_transformed_data,
                    )
                    for i in range(self._n_jobs)
                )

                estimators, transformed_data = zip(*fit)

                self.estimators_ += estimators
                Xt += transformed_data

                self.n_estimators_ += self._n_jobs
                train_time = time.time() - start_time

                if log_each_estimator:
                    contract_remaining = self._format_duration(
                        max(0.0, time_limit - train_time)
                    )
                    first_estimator = self.n_estimators_ - len(fit) + 1
                    for estimator_idx in range(first_estimator, self.n_estimators_ + 1):
                        self._log(
                            f"[Arsenal] Estimator {estimator_idx}: "
                            f"elapsed={train_time:.2f}s, "
                            f"contract_remaining={contract_remaining}"
                        )
                elif log_progress and train_time >= next_progress:
                    self._log(
                        f"[Arsenal] Progress: built={self.n_estimators_}, "
                        f"elapsed={train_time:.2f}s"
                    )
                    next_progress = train_time + progress_interval
        else:
            if self.verbose > 0:
                estimator_start_time = time.time()
                if log_each_estimator:
                    batch_size = self._n_jobs
                else:
                    batch_size = max(self._n_jobs, (self.n_estimators + 9) // 10)

                fit = []
                for batch_start in range(0, self.n_estimators, batch_size):
                    current_batch_size = min(
                        batch_size, self.n_estimators - batch_start
                    )
                    batch_fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                        delayed(self._fit_ensemble_estimator)(
                            _clone_estimator(
                                base_rocket, rng.randint(np.iinfo(np.int32).max)
                            ),
                            X,
                            y,
                            keep_transformed_data=keep_transformed_data,
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
                                self.n_estimators - built
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
                                f"[Arsenal] Estimator "
                                f"{estimator_idx}/{self.n_estimators}: "
                                f"elapsed={elapsed:.2f}s, {time_estimate}"
                            )
                    else:
                        self._log(
                            f"[Arsenal] Progress: built={built}/{self.n_estimators}, "
                            f"elapsed={time.time() - start_time:.2f}s"
                        )
            else:
                fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                    delayed(self._fit_ensemble_estimator)(
                        _clone_estimator(
                            base_rocket, rng.randint(np.iinfo(np.int32).max)
                        ),
                        X,
                        y,
                        keep_transformed_data=keep_transformed_data,
                    )
                    for _ in range(self.n_estimators)
                )

            self.estimators_, Xt = zip(*fit)
            self.n_estimators_ = self.n_estimators

        self.weights_ = []
        self._weight_sum = 0
        for rocket_pipeline in self.estimators_:
            weight = rocket_pipeline.steps[2][1].best_score_
            self.weights_.append(weight)
            self._weight_sum += weight

        if self.verbose > 0:
            self._log(
                f"[Arsenal] Finished fit: built={self.n_estimators_}, "
                f"elapsed={time.time() - start_time:.2f}s"
            )

        return Xt

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

    def _fit_ensemble_estimator(self, rocket, X, y, keep_transformed_data):
        transformed_x = rocket.fit_transform(X)
        scaler = StandardScaler(with_mean=False)
        scaler.fit(transformed_x, y)
        ridge = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
        )
        ridge.fit(scaler.transform(transformed_x), y)
        return [
            make_pipeline(rocket, scaler, ridge),
            transformed_x if keep_transformed_data else None,
        ]

    def _predict_proba_for_estimator(self, X, classifier, idx):
        preds = classifier.predict(X)
        weights = np.zeros((X.shape[0], self.n_classes_))
        for i in range(X.shape[0]):
            weights[i, self._class_dictionary[preds[i]]] += self.weights_[idx]
        return weights

    def _train_probas_for_estimator(self, Xt, y, idx, rng):
        indices = range(self.n_cases_)
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = [n for n in indices if n not in subsample]

        results = np.zeros((self.n_cases_, self.n_classes_))
        if not oob:
            return results, 1, oob

        clf = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
            ),
        )
        clf.fit(Xt[idx][subsample], y[subsample])
        preds = clf.predict(Xt[idx][oob])

        weight = clf.steps[1][1].best_score_

        for n, pred in enumerate(preds):
            results[oob[n]][self._class_dictionary[pred]] += weight

        return results, weight, oob

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Arsenal provides the following special sets:
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
        if parameter_set == "results_comparison":
            return {"n_kernels": 20, "n_estimators": 5}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "n_kernels": 10,
                "contract_max_n_estimators": 2,
            }
        else:
            return {"n_kernels": 10, "n_estimators": 2}
