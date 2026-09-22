"""Arsenal classifier.

kernel based ensemble of ROCKET classifiers.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["Arsenal"]

import time
from copy import deepcopy

import numpy as np
from joblib import delayed
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from aeon.base import CheckpointableMixin
from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MultiRocket,
    Rocket,
)
from aeon.utils._parallel import _run_jobs
from aeon.utils.validation import check_n_jobs


def _get_oob_indices(subsample, n_cases):
    """Return ordered out-of-bag indices for a bootstrap sample."""
    in_bag = np.zeros(n_cases, dtype=bool)
    in_bag[subsample] = True
    return np.flatnonzero(~in_bag)


_ALPHAS = np.logspace(-3, 3, 10)


def _fit_ridge_classifier(X, y, class_weight):
    """Fit a member ridge whose ``best_score_`` is its LOO CV accuracy.

    For two classes ``RidgeClassifierCV`` reconstructs leave-one-out labels with
    ``argmax`` over a single signed column, so ``best_score_`` is always 1.0 and
    the first alpha is always chosen. See
    https://github.com/scikit-learn/scikit-learn/issues/34942. Binary problems
    therefore take the predicted class from the sign of the stored leave-one-out
    decision values and refit at the best alpha. Multiclass is unchanged.
    """
    ridge = RidgeClassifierCV(
        alphas=_ALPHAS,
        class_weight=class_weight,
        scoring="accuracy",
        store_cv_results=len(np.unique(y)) == 2,
    ).fit(X, y)
    cv_results = getattr(ridge, "cv_results_", None)
    if cv_results is None or cv_results.ndim != 3 or cv_results.shape[1] != 1:
        return ridge

    positive = (y == ridge.classes_[1])[:, None]
    accuracies = ((cv_results[:, 0, :] > 0) == positive).mean(axis=0)
    best = int(np.argmax(accuracies))
    binary = RidgeClassifier(alpha=_ALPHAS[best], class_weight=class_weight).fit(X, y)
    binary.best_score_ = float(accuracies[best])
    binary.alpha_ = _ALPHAS[best]
    return binary


def _transform_with(rocket, X, pre_normalised):
    """Apply a fitted rocket transform to already-validated input.

    ``X`` has been validated by the ensemble, so the private transform is used
    either way. When the ensemble has already normalised ``X`` (``Rocket``
    only, whose ``normalise`` defaults to True), the kernels-only
    ``_transform_kernels`` is called so the series are not normalised once per
    ensemble member. MiniRocket never normalises and MultiRocket defaults to
    ``normalise=False``, so for those ``_transform`` is already kernels-only.
    """
    if pre_normalised:
        return rocket._transform_kernels(X)
    return rocket._transform(X)


def _normalise_oob_probabilities(probabilities, weights, oobs, n_classes):
    """Normalize summed OOB probabilities by each case's available weight."""
    divisors = np.zeros(probabilities.shape[0])
    for weight, oob in zip(weights, oobs):
        divisors[oob] += weight

    has_predictions = divisors != 0
    probabilities[~has_predictions] = 1 / n_classes
    probabilities[has_predictions] /= divisors[has_predictions, None]
    return probabilities


def _aggregate_class_votes(class_indices, weights, n_cases, n_classes, oobs=None):
    """Aggregate weighted class-index votes into a probability matrix."""
    probabilities = np.zeros((n_cases, n_classes))
    if oobs is None:
        case_indices = np.arange(n_cases)
        for predictions, weight in zip(class_indices, weights):
            probabilities[case_indices, predictions] += weight
    else:
        for predictions, weight, oob in zip(class_indices, weights, oobs):
            probabilities[oob, predictions] += weight
    return probabilities


class Arsenal(CheckpointableMixin, BaseClassifier):
    """
    Arsenal ensemble.

    Overview: an ensemble of ROCKET transformers using RidgeClassifierCV base
    classifier. Weights each classifier using the accuracy from the ridge
    cross-validation. Allows for generation of probability estimates at the
    expense of scalability compared to RocketClassifier.

    Parameters
    ----------
    n_kernels : int, default=2,000
        Number of kernels for each ROCKET transform.
    n_estimators : int, default=25
        Number of estimators to build for the ensemble.
    rocket_transform : str, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket","minirocket","multirocket"].
    max_dilations_per_kernel : int, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=4
        MultiRocket only. The number of features per kernel.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used. Each call to ``fit`` or
        ``resume_fit`` receives a new time budget. A running batch may overrun it.
    contract_max_n_estimators : int, default=100
        Max number of estimators when time_limit_in_minutes is set.
    class_weight : dict or "balanced", default=None
        The ``class_weight`` passed to each ensemble member's
        ``RidgeClassifierCV``. If None, all classes have weight one. The
        "balanced" mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data, as
        ``n_samples / (n_classes * np.bincount(y))``. A dict maps class
        labels to weights.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
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
    checkpoint_path : str, pathlib.Path or None, default=None
        Checkpoint file, saved at completed batch boundaries and on successful
        completion. None disables automatic writes. The parent must exist.
    checkpoint_interval : float or None, default=None
        Minimum minutes between periodic checkpoint writes. None saves only on
        successful completion when a path is configured.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of channels per case.
    n_timepoints_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    weights_ : list of shape (n_estimators) of float
        Weight of each estimator in the ensemble.
    n_estimators_ : int
        The number of estimators in the ensemble.
    fit_elapsed_time_ : float
        Accumulated fitting time in seconds across completed batches and calls.

    See Also
    --------
    RocketClassifier
        Arsenal is an ensemble of RocketClassifier.

    Notes
    -----
    ``resume_fit(X, y)`` continues from saved state using the original training
    data. ``fit`` always starts afresh. Between calls, only ``n_estimators``,
    ``contract_max_n_estimators``, ``time_limit_in_minutes``, ``n_jobs``,
    ``verbose`` and checkpoint settings may change. Member limits apply to the
    whole ensemble, including existing members. Training estimates, when
    requested in the original fit, are retained in continuation state.

    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/kernel_based/Arsenal.java>`_.

    References
    ----------
    .. [1] Middlehurst, M., Large, J., Flynn, M. et al.
       HIVE-COTE 2.0: a new meta ensemble for time series classification.
       Mach Learn 110, 3211–3243 (2021).
       https://doi.org/10.1007/s10994-021-06057-9

    Examples
    --------
    >>> from aeon.classification.convolution_based import Arsenal
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test =load_unit_test(split="test")
    >>> clf = Arsenal(n_kernels=100, n_estimators=5)
    >>> clf.fit(X_train, y_train)
    Arsenal(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:checkpointing": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    _checkpoint_mutable_params = CheckpointableMixin._checkpoint_mutable_params + (
        "n_estimators",
        "contract_max_n_estimators",
        "time_limit_in_minutes",
    )

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
        checkpoint_path=None,
        checkpoint_interval=None,
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
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval

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
        rng = deepcopy(check_random_state(self.random_state))
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
        if self.rocket_transform == "rocket":
            X = Normalizer().fit_transform(X).astype(np.float32, copy=False)

        y_probas = _run_jobs(
            (
                delayed(self._predict_for_estimator)(
                    X,
                    self.estimators_[i],
                )
                for i in range(self.n_estimators_)
            ),
            self._n_jobs,
            prefer="threads",
        )

        probabilities = _aggregate_class_votes(
            y_probas,
            self.weights_,
            X.shape[0],
            self.n_classes_,
        )
        return probabilities / self._weight_sum

    def _fit_predict(self, X, y) -> np.ndarray:
        rng = deepcopy(check_random_state(self.random_state))
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._fit_predict_proba(X, y)
            ]
        )

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        train_estimates = self._fit_arsenal(X, y, return_train_estimates=True)
        class_indices, weights, oobs = zip(*train_estimates)

        results = _aggregate_class_votes(
            class_indices,
            weights,
            self.n_cases_,
            self.n_classes_,
            oobs,
        )
        return _normalise_oob_probabilities(
            results,
            weights,
            oobs,
            self.n_classes_,
        )

    def _fit_arsenal(self, X, y, return_train_estimates=False):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        if self.rocket_transform == "rocket":
            self._base_rocket = Rocket(n_kernels=self.n_kernels)
        elif self.rocket_transform == "minirocket":
            self._base_rocket = MiniRocket(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
            )
        elif self.rocket_transform == "multirocket":
            self._base_rocket = MultiRocket(
                n_kernels=self.n_kernels,
                max_dilations_per_kernel=self.max_dilations_per_kernel,
                n_features_per_kernel=self.n_features_per_kernel,
            )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")

        self._rng = check_random_state(self.random_state)
        self._train_rng = (
            check_random_state(self.random_state) if return_train_estimates else None
        )
        self.estimators_ = []
        self.weights_ = []
        self._train_estimates = []
        self.n_estimators_ = 0
        self.fit_elapsed_time_ = 0.0
        self._continue_arsenal(X, y)
        return self._train_estimates if return_train_estimates else None

    def _resume_fit(self, X, y):
        self._continue_arsenal(X, y)
        return self

    def _continue_arsenal(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)
        time_limit = self.time_limit_in_minutes * 60
        target = self.contract_max_n_estimators if time_limit > 0 else self.n_estimators
        if not isinstance(target, (int, np.integer)) or target < 1:
            raise ValueError(
                "The target number of estimators must be a positive integer."
            )
        start_time = time.perf_counter()
        train_time = 0.0
        initial_count = self.n_estimators_
        if self.rocket_transform == "rocket":
            X = Normalizer().fit_transform(X).astype(np.float32, copy=False)

        log_each_estimator = self.verbose >= 2
        log_progress = self.verbose == 1
        progress_interval = time_limit / 10 if time_limit > 0 else 0
        next_progress = progress_interval
        if self.verbose > 0:
            fit_limit = (
                f"time_limit={self._format_duration(time_limit)}, "
                f"max_n_estimators={target}"
                if time_limit > 0
                else f"n_estimators={target}"
            )
            self._log(
                f"[{type(self).__name__}] Starting fit: n_cases={self.n_cases_}, "
                f"n_channels={self.n_channels_}, n_timepoints={self.n_timepoints_}, "
                f"transform={self.rocket_transform}, n_kernels={self.n_kernels}, "
                f"{fit_limit}, n_jobs={self._n_jobs}"
            )

        batch_size = (
            self._n_jobs
            if time_limit > 0 or self.checkpoint_path is not None
            else max(self._n_jobs, (target + 9) // 10) if self.verbose > 0 else target
        )
        while self.n_estimators_ < target and (
            time_limit <= 0 or train_time < time_limit
        ):
            current_batch_size = min(batch_size, target - self.n_estimators_)
            rng, train_rng = deepcopy((self._rng, self._train_rng))
            self._checkpoint_ready = False
            fit = _run_jobs(
                (
                    delayed(self._fit_ensemble_estimator)(
                        _clone_estimator(
                            self._base_rocket, rng.randint(np.iinfo(np.int32).max)
                        ),
                        X,
                        y,
                        train_rng=(
                            check_random_state(
                                train_rng.randint(np.iinfo(np.int32).max)
                            )
                            if train_rng is not None
                            else None
                        ),
                    )
                    for _ in range(current_batch_size)
                ),
                self._n_jobs,
                prefer="threads",
            )
            estimators, weights, train_data = zip(*fit)
            self.estimators_.extend(estimators)
            self.weights_.extend(weights)
            if self._train_rng is not None:
                self._train_estimates.extend(train_data)
            self._rng.set_state(rng.get_state())
            if self._train_rng is not None:
                self._train_rng.set_state(train_rng.get_state())
            self.n_estimators_ = len(self.estimators_)
            self._weight_sum = float(np.sum(self.weights_))
            elapsed = time.perf_counter() - start_time
            self.fit_elapsed_time_ += elapsed - train_time
            train_time = elapsed
            self._checkpoint_parameter_signature = self._checkpoint_parameter_hash()
            self._checkpoint_ready = True
            self._maybe_checkpoint()

            if log_each_estimator:
                if time_limit > 0:
                    remaining = (
                        "contract_remaining="
                        f"{self._format_duration(max(0.0, time_limit - train_time))}"
                    )
                else:
                    estimate = train_time / (self.n_estimators_ - initial_count)
                    remaining_time = estimate * (target - self.n_estimators_)
                    remaining = (
                        "estimated_remaining="
                        f"{self._format_duration(remaining_time)}"
                    )
                for estimator_idx in range(
                    self.n_estimators_ - current_batch_size + 1, self.n_estimators_ + 1
                ):
                    member = (
                        str(estimator_idx)
                        if time_limit > 0
                        else f"{estimator_idx}/{target}"
                    )
                    self._log(
                        f"[{type(self).__name__}] Estimator {member}: "
                        f"elapsed={train_time:.2f}s, {remaining}"
                    )
            elif log_progress and train_time >= next_progress:
                built = (
                    str(self.n_estimators_)
                    if time_limit > 0
                    else f"{self.n_estimators_}/{target}"
                )
                self._log(
                    f"[{type(self).__name__}] Progress: built={built}, "
                    f"elapsed={train_time:.2f}s"
                )
                next_progress = train_time + progress_interval

            elapsed = time.perf_counter() - start_time
            self.fit_elapsed_time_ += elapsed - train_time
            train_time = elapsed

        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] Finished fit: built={self.n_estimators_}, "
                f"elapsed={time.perf_counter() - start_time:.2f}s"
            )

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

    def _fit_ensemble_estimator(self, rocket, X, y, train_rng=None):
        # X is already normalised at ensemble level where the transformer
        # needs it, so kernels are applied without further preprocessing
        rocket.fit(X)
        transformed_x = _transform_with(rocket, X, self.rocket_transform == "rocket")
        scaler = StandardScaler(with_mean=False)
        # best_score_ is the LOO CV accuracy used to weight this member
        ridge = _fit_ridge_classifier(
            scaler.fit_transform(transformed_x), y, self.class_weight
        )
        pipeline = make_pipeline(rocket, scaler, ridge)

        train_estimate = (
            self._train_probas_for_estimator(transformed_x, y, train_rng)
            if train_rng is not None
            else None
        )
        return pipeline, ridge.best_score_, train_estimate

    def _predict_for_estimator(self, X, classifier):
        rocket, scaler, ridge = (step[1] for step in classifier.steps)
        transformed_x = _transform_with(rocket, X, self.rocket_transform == "rocket")
        preds = ridge.predict(scaler.transform(transformed_x))
        return np.searchsorted(self.classes_, preds)

    def _train_probas_for_estimator(self, Xt, y, rng):
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = _get_oob_indices(subsample, self.n_cases_)

        if oob.size == 0:
            # no out-of-bag cases: the member contributes no train estimates,
            # so its weight is zero evidence rather than a fake accuracy
            return np.empty(0, dtype=np.intp), 0.0, oob

        scaler = StandardScaler(with_mean=False)
        ridge = _fit_ridge_classifier(
            scaler.fit_transform(Xt[subsample]), y[subsample], self.class_weight
        )
        preds = ridge.predict(scaler.transform(Xt[oob]))

        return np.searchsorted(self.classes_, preds), ridge.best_score_, oob

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
        elif parameter_set == "checkpointing":
            return {"n_kernels": 10, "n_estimators": 3}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "n_kernels": 10,
                "contract_max_n_estimators": 2,
            }
        else:
            return {"n_kernels": 10, "n_estimators": 2}
