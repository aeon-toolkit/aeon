"""
implement for imblearn minority class rebalancer ADASYN.
see more in imblearn.over_sampling.ADASYN
original authors:
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT
"""
import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scipy import sparse
from collections import OrderedDict

__maintainer__ = ["TonyBagnall, Chris Qiu"]
__all__ = ["ADASYN"]


class ADASYN(BaseCollectionTransformer):
    """
    Class to perform over-sampling using ADASYN .

    This object is a simplified implementation of ADASYN - Adaptive
    Synthetic (ADASYN) algorithm as presented in imblearn.over_sampling.ADASYN
    This method is similar to SMOTE, but it generates different number of
    samples depending on an estimate of the local distribution of the class
    to be oversampled.
    Parameters
    ----------
    {random_state}

    k_neighbors : int or object, default=5
        The nearest neighbors used to define the neighborhood of samples to use
        to generate the synthetic samples. `~sklearn.neighbors.NearestNeighbors`
        instance will be fitted in this case.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "requires_y": True,
    }

    def __init__(self, random_state=None, k_neighbors=5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        super().__init__()

    def _fit(self, X, y=None):
        # set the additional_neighbor=1
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors + 1)

        # generate sampling target by targeting all classes but not the majority
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(
            sorted(sampling_strategy.items())
        )
        return self

    def _transform(self, X, y=None):
        shape_recover = False  # use to recover the shape of X
        if X.ndim == 3 and X.shape[1] == 1:
            X = np.squeeze(X, axis=1)  # remove the middle dimension to be compatible with sklearn
            shape_recover = True
        random_state = check_random_state(self.random_state)
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated i.e. num_majority - num_minority
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]

            self.nn_.fit(X)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.
            n_neighbors = self.nn_.n_neighbors - 1
            ratio_nn = np.sum(y[nns] != class_sample, axis=1) / n_neighbors
            if not np.sum(ratio_nn):
                raise RuntimeError(
                    "Not any neigbours belong to the majority"
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
            self.nn_.fit(X_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]

            enumerated_class_indices = np.arange(len(target_class_indices))
            rows = np.repeat(enumerated_class_indices, n_samples_generate)
            cols = random_state.choice(n_neighbors, size=n_samples)
            diffs = X_class[nns[rows, cols]] - X_class[rows]
            steps = random_state.uniform(size=(n_samples, 1))

            if sparse.issparse(X):
                sparse_func = type(X).__name__
                steps = getattr(sparse, sparse_func)(steps)
                X_new = X_class[rows] + steps.multiply(diffs)
            else:
                X_new = X_class[rows] + steps * diffs

            X_new = X_new.astype(X.dtype)
            y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        if shape_recover:
            X_resampled = X_resampled[:, np.newaxis, :]
        return X_resampled, y_resampled
