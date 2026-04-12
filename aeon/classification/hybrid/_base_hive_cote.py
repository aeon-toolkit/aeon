"""Base class for Hierarchical Vote Collective of Transformation-based Ensembles."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.utils.validation import check_n_jobs


class BaseHIVECOTE(BaseClassifier):
    """
    Modular Base class for HIVE-COTE ensembles.

    This class handles the core logic of the CAWPE (Cross-validation Accuracy
    Weighted Probabilistic Ensemble) structure. It accepts a list of base
    estimators, trains them using fit_predict to get out-of-bag estimates,
    calculates their weights based on training accuracy, and combines their
    probabilities during prediction.

    Parameters
    ----------
    estimators : list of tuples
        List of (name, estimator) tuples representing the ensemble components.
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
        "capability:multivariate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        estimators=None,
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

        # Internal state variables
        self.fitted_estimators_ = []
        self.weights_ = []
        self.component_names_ = []

        super().__init__()

    def _fit(self, X, y):
        """Fit the ensemble to training data and calculate CAWPE weights."""
        self._n_jobs = check_n_jobs(self.n_jobs)

        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError("No estimators provided to BaseHIVECOTE.")

        self.fitted_estimators_ = []
        self.weights_ = []
        self.component_names_ = []

        # Dynamically traverse and train all the underlying components
        for name, estimator in self.estimators:
            # 1. Clone model to prevent mutating the original instance
            est = clone(estimator)

            # 2. Pass global parameters
            if hasattr(est, "random_state") and self.random_state is not None:
                est.random_state = self.random_state
            if hasattr(est, "n_jobs") and self.n_jobs is not None:
                est.n_jobs = self._n_jobs

            # 3. Call fit_predict to get internal cross-validation/OOB predictions
            train_preds = est.fit_predict(X, y)

            # 4. Calculate CAWPE weight: accuracy ^ alpha
            weight = accuracy_score(y, train_preds) ** self.alpha

            # 5. Store state
            self.fitted_estimators_.append(est)
            self.weights_.append(weight)
            self.component_names_.append(name)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class labels for X."""
        # Predict based on the final probabilities calculated by _predict_proba
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

        # Multiply each component's probabilities by
        # its corresponding weight and accumulate
        for i, est in enumerate(self.fitted_estimators_):
            weight = self.weights_[i]
            probas = est.predict_proba(X)
            dists = np.add(dists, probas * weight)

        sums = dists.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        # Normalize to ensure probabilities sum to 1 for each instance
        return dists / sums

    def get_component_weights(self):
        """Return the calculated weights for each component."""
        return dict(zip(self.component_names_, self.weights_))
