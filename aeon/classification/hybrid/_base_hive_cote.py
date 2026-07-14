"""Base class for Hierarchical Vote Collective of Transformation-based Ensembles."""

from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.utils.validation import check_n_jobs


class _BaseHIVECOTE(BaseClassifier):
    """
    Modular base class for HIVE-COTE ensembles.

    This class handles the core logic of the CAWPE (Cross-validation Accuracy
    Weighted Probabilistic Ensemble) structure. It accepts a list of base
    estimators, trains them using fit_predict to get out-of-bag estimates,
    calculates their weights based on training accuracy, and combines their
    probabilities during prediction.

    Parameters
    ----------
    estimators : list of tuples
        List of (name, estimator) tuples representing the ensemble components.
        Each estimator must be an instance of BaseClassifier.
    alpha : int or float, default=4
        The power parameter for the CAWPE weight calculation.
    random_state : int, RandomState instance or None, default=None
        Seed for random number generation.
    n_jobs : int, default=1
        The number of jobs to run in parallel.
    verbose : int, default=0
        Level of output printed to the console.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:contractable": False,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        estimators,
        alpha=4,
        random_state=None,
        n_jobs=1,
        verbose=0,
    ):
        self.estimators = estimators
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        super().__init__()

    def _fit(self, X, y):
        """Fit the ensemble to training data and calculate CAWPE weights."""
        self._n_jobs = check_n_jobs(self.n_jobs)
        verbose_name = getattr(self, "_verbose_name", None)
        logging_enabled = verbose_name is not None and self.verbose > 0
        total_start = perf_counter() if logging_enabled else None

        if logging_enabled:
            self._log(
                f"[{verbose_name}] Starting fit: n_cases={X.shape[0]}, "
                f"n_channels={X.shape[1]}, n_timepoints={X.shape[2]}, "
                f"n_jobs={self._n_jobs}"
            )
            self._log_fit_configuration()

        # Subclasses may construct their estimator list during fit and store it
        # in self._estimators to avoid mutating the init parameter self.estimators
        # (required for scikit-learn compatibility: __init__ parameters must not
        # be modified by fit).
        estimators = getattr(self, "_estimators", None)
        if estimators is None:
            estimators = self.estimators

        if estimators is None or len(estimators) == 0:
            raise ValueError("No estimators provided to _BaseHIVECOTE.")

        for name, estimator in estimators:
            if not isinstance(estimator, BaseClassifier):
                raise TypeError(
                    f"Estimator '{name}' is not a BaseClassifier instance. "
                    "All components must be aeon classifiers."
                )

        # Reset fitted state to avoid accumulation on re-fit
        self.fitted_estimators_ = []
        self.weights_ = []
        self.component_names_ = []
        component_summaries = []

        # Dynamically traverse and train all the underlying components
        for name, estimator in estimators:
            est = clone(estimator)

            # Pass global parameters to the cloned estimator
            if hasattr(est, "random_state") and self.random_state is not None:
                est.random_state = self.random_state
            if hasattr(est, "n_jobs"):
                est.n_jobs = self._n_jobs
            if hasattr(est, "verbose"):
                est.verbose = (
                    max(0, self.verbose - 2)
                    if verbose_name is not None
                    else self.verbose
                )

            if logging_enabled:
                self._log(f"[{verbose_name}] Starting {name}...")
                if self.verbose >= 2:
                    self._log(
                        f"[{verbose_name}] {name} params: "
                        f"{est.get_params(deep=False)}",
                        level=2,
                    )

            # Get OOB/CV predictions and calculate CAWPE weight
            component_start = perf_counter() if logging_enabled else None
            train_preds = est.fit_predict(X, y)
            train_acc = accuracy_score(y, train_preds)
            weight = train_acc**self.alpha

            if logging_enabled:
                component_elapsed = perf_counter() - component_start
                self._log(
                    f"[{verbose_name}] Finished {name} in {component_elapsed:.2f}s, "
                    f"train_acc={train_acc:.4f}, weight={weight:.4f}"
                )
                component_summaries.append(
                    f"{name}(train_acc={train_acc:.4f}, weight={weight:.4f})"
                )

            self.fitted_estimators_.append(est)
            self.weights_.append(weight)
            self.component_names_.append(name)

        if logging_enabled:
            total_elapsed = perf_counter() - total_start
            self._log(f"[{verbose_name}] Finished fit in {total_elapsed:.2f}s")
            self._log(
                f"[{verbose_name}] Component summary: " + ", ".join(component_summaries)
            )

        return self

    def _log(self, message, level=1):
        """Print a message when the configured verbosity reaches ``level``."""
        if self.verbose >= level:
            print(message, flush=True)  # noqa: T201

    def _log_fit_configuration(self):
        """Log subclass-specific fit configuration when verbosity is enabled."""

    def _predict(self, X) -> np.ndarray:
        """Predict class labels for X."""
        dists = self._predict_proba(X)
        rng = check_random_state(self.random_state)

        preds = [
            self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
            for prob in dists
        ]
        return np.array(preds, dtype=self.classes_.dtype)

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X using CAWPE."""
        dists = np.zeros((X.shape[0], self.n_classes_))

        for i, est in enumerate(self.fitted_estimators_):
            weight = self.weights_[i]
            probas = est.predict_proba(X)
            dists = np.add(dists, probas * weight)

        sums = dists.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return dists / sums

    def get_component_weights(self):
        """Return the calculated weights for each component."""
        return dict(
            zip(
                getattr(self, "component_names_", []),
                getattr(self, "weights_", []),
            )
        )
