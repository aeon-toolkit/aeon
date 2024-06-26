"""Proximity Forest Classifier.

The Proximity Forest is an ensemble of Proximity Trees.
"""

from typing import Type, Union

import numpy as np

from aeon.classification.base import BaseClassifier
from aeon.classification.distance_based import ProximityTree


class ProximityForest(BaseClassifier):
    """Proximity Forest Classifier.

    The Proximity Forest is an ensemble of Proximity Trees.
    """

    def __init__(
        self,
        n_trees=10,
        n_splitters: int = 5,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: Union[int, Type[np.random.RandomState], None] = None,
        n_jobs: int = 1,
    ):
        self.n_trees = n_trees
        self.n_splitters = n_splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y):
        # Check dimension of X
        if X.ndim == 3:
            if X.shape[1] == 1:
                X = np.squeeze(X, axis=1)
            else:
                raise ValueError("X should be univariate.")

        self.classes_ = list(np.unique(y))
        self.trees_ = []
        for _ in range(self.n_trees):
            clf = ProximityTree(
                n_splitters=self.n_splitters,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            clf.fit(X, y)
            self.trees_.append(clf)

        self._is_fitted = True

    def _predict_proba(self, X):
        # Check dimension of X
        if X.ndim == 3:
            if X.shape[1] == 1:
                X = np.squeeze(X, axis=1)
            else:
                raise ValueError("X should be univariate.")

        output_probas = []
        for i in range(self.n_trees):
            proba = self.trees_[i].predict_proba(X)
            output_probas.append(proba)

        output_probas = np.sum(output_probas, axis=0)
        output_probas = np.divide(output_probas, self.n_trees)
        return output_probas

    def _predict(self, X):
        probas = self._predict_proba(X)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds
