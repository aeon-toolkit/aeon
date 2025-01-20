"""
implement for imblearn minority class rebalancer SMOTE.
see more in imblearn.over_sampling.SMOTE
original authors:
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT
"""

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scipy import sparse
from collections import OrderedDict

__maintainer__ = ["TonyBagnall, Chris Qiu"]
__all__ = ["SMOTE"]


class SMOTE(BaseCollectionTransformer):
    """
    Class to perform over-sampling using SMOTE.

    This object is a simplified implementation of SMOTE - Synthetic Minority
    Over-sampling Technique as presented in imblearn.over_sampling.SMOTE
    sampling_strategy is sampling target by targeting all classes but not the
    majority, which directly expressed in _fit.sampling_strategy.
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
        shape_recover = False   # use to recover the shape of X
        if X.ndim == 3 and X.shape[1] == 1:
            X = np.squeeze(X, axis=1)  # remove the middle dimension to be compatible with sklearn
            shape_recover = True
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated i.e. num_majority - num_minority
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]

            self.nn_.fit(X_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
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

    def _make_samples(
            self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, y=None
    ):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        y : ndarray of shape (n_samples_all,), default=None
            The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
            weight the distances in the sample generation process.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_samples(
            self, X, nn_data, nn_num, rows, cols, steps, y_type=None, y=None
    ):
        r"""Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times
           (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is
        the current sample, \mathbf{s_{nn}} is a randomly selected neighbors of
        \mathbf{s_{i}} and \mathcal{u}(0, 1) is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        rows : ndarray of shape (n_samples,), dtype=int
            Indices pointing at feature vector in X which will be used
            as a base for creating new samples.

        cols : ndarray of shape (n_samples,), dtype=int
            Indices pointing at which nearest neighbor of base feature vector
            will be used when creating new samples.

        steps : ndarray of shape (n_samples,), dtype=float
            Step sizes for new samples.

        y_type : str, int or None, default=None
            Class label of the current target classes for which we want to generate
            samples.

        y : ndarray of shape (n_samples_all,), default=None
            The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
            weight the distances in the sample generation process.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Synthetically generated samples.
        """
        diffs = nn_data[nn_num[rows, cols]] - X[rows]
        if y is not None:  # only entering for BorderlineSMOTE-2
            random_state = check_random_state(self.random_state)
            mask_pair_samples = y[nn_num[rows, cols]] != y_type
            diffs[mask_pair_samples] *= random_state.uniform(
                low=0.0, high=0.5, size=(mask_pair_samples.sum(), 1)
            )

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs

        return X_new.astype(X.dtype)



