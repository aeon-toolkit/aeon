"""ADASYN over sampling algorithm.

See more in imblearn.over_sampling.ADASYN
original authors:
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["ADASYN"]

from collections.abc import Callable

import numpy as np

from aeon.transformations.collection.imbalance._smote import SMOTE


class ADASYN(SMOTE):
    """
    Adaptive Synthetic Sampling (ADASYN) over-sampler.

    Generates synthetic samples for the minority class based on local data
    distribution. ADASYN extends SMOTE by adapting the number of synthetic samples
    according to the density of the minority class: more samples are generated for
    minority samples that are harder to learn (i.e., surrounded by more majority
    samples).

    This implementation is adapted from imbalanced-learn's
    `imblearn.over_sampling.ADASYN`.

    Parameters
    ----------
        random_state : int or None, optional (default=None)
            Random seed for reproducibility.
        n_neighbors : int, optional (default=5)
            Number of nearest neighbours used to construct synthetic samples.

    References
    ----------
    .. [1] He, H., Bai, Y., Garcia, E. A., & Li, S. (2008).
           ADASYN: Adaptive synthetic sampling approach for imbalanced learning.
           In IEEE International Joint Conference on Neural Networks, pp. 1322-1328.
           https://doi.org/10.1109/IJCNN.2008.4633969

    Examples
    --------
    >>> from aeon.transformations.collection.imbalance import ADASYN
    >>> import numpy as np
    >>> X = np.random.random(size=(100,1,50))
    >>> y = np.array([0] * 90 + [1] * 10)
    >>> sampler = ADASYN(random_state=49)
    >>> X_res, y_res = sampler.fit_transform(X, y)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        random_state=None,
        distance: str | Callable = "euclidean",
        distance_params: dict | None = None,
        n_jobs: int = 1,
        weights: str | Callable = "uniform",
    ):
        super().__init__(
            random_state=random_state,
            n_neighbors=n_neighbors,
            distance=distance,
            distance_params=distance_params,
            n_jobs=n_jobs,
            weights=weights,
        )

    def _transform(self, X, y=None):
        X = np.squeeze(X, axis=1)
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            y_class = y[target_class_indices]

            self.nn_.fit(X, y)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.
            n_neighbors = self.nn_.n_neighbors - 1
            ratio_nn = np.sum(y[nns] != class_sample, axis=1) / n_neighbors
            if not np.sum(ratio_nn):
                raise RuntimeError(
                    "Not any neighbours belong to the majority"
                    " class. This case will induce a NaN case"
                    " with a division by zero. ADASYN is not"
                    " suited for this specific dataset."
                    " Use SMOTE instead."
                )
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)
            # rounding may cause new amount for n_samples
            n_samples = np.sum(n_samples_generate)
            if not n_samples:
                raise ValueError(
                    "No samples will be generated with the provided ratio settings."
                )

            # the nearest neighbors need to be fitted only on the current class
            # to find the class NN to generate new samples
            self.nn_.fit(X_class, y_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]

            enumerated_class_indices = np.arange(len(target_class_indices))
            rows = np.repeat(enumerated_class_indices, n_samples_generate)
            cols = self._random_state.choice(n_neighbors, size=n_samples)
            diffs = X_class[nns[rows, cols]] - X_class[rows]
            steps = self._random_state.uniform(size=(n_samples, 1))
            X_new = X_class[rows] + steps * diffs

            X_new = X_new.astype(X.dtype)
            y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
            X_resampled.append(X_new)
            y_resampled.append(y_new)
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        X_resampled = X_resampled[:, np.newaxis, :]
        return X_resampled, y_resampled
