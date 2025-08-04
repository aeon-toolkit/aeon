"""SMOTE over sampling algorithm.

See more in imblearn.over_sampling.SMOTE
original authors:
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT
"""

from collections import OrderedDict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["TonyBagnall"]
__all__ = ["SMOTE"]


class SMOTE(BaseCollectionTransformer):
    """
    Synthetic Minority Over-sampling TEchnique (SMOTE) for imbalanced datasets.

    Generates synthetic samples of the minority class to address class imbalance.
    SMOTE constructs new samples by interpolating between existing minority samples
    and their nearest neighbours in feature space.

    This implementation adapts the algorithm from `imblearn.over_sampling.SMOTE`.
    It targets all classes except the majority, as controlled by the `sampling_strategy`
    in the `_fit` method. It uses ``aeon`` distances to find the nearest neighbours.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbours used to generate synthetic samples. A
        `sklearn.neighbors.NearestNeighbors` instance is fitted for this purpose.
    random_state : int, RandomState instance or None, default=None
        Controls the random number generation for reproducibility:
        - If `int`, sets the random seed.
        - If `RandomState` instance, uses it as the generator.
        - If `None`, uses `np.random`.

    See Also
    --------
    ADASYN : Adaptive synthetic sampling extension to SMOTE.

    References
    ----------
    .. [1] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
           SMOTE: Synthetic minority over-sampling technique.
           Journal of Artificial Intelligence Research, 16, 321–357.
           https://dl.acm.org/doi/10.5555/1622407.1622416

    Examples
    --------
    >>> from aeon.transformations.collection.imbalance import SMOTE
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> import numpy as np
    >>> X = make_example_3d_numpy(n_cases=100, return_y=False, random_state=49)
    >>> y = np.array([0] * 90 + [1] * 10)
    >>> sampler = SMOTE(random_state=49)
    >>> X_res, y_res = sampler.fit_transform(X, y)
    >>> y_res.shape
    (180,)
    """

    _tags = {
        "requires_y": True,
    }

    def __init__(self, k_neighbors: int = 5, random_state=None):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        super().__init__()

    def _fit(self, X, y=None):
        # set the additional_neighbor required by SMOTE
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors + 1)

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))
        return self

    def _transform(self, X, y=None):
        # remove the channel dimension to be compatible with sklearn
        X = np.squeeze(X, axis=1)
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated
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
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        X_resampled = X_resampled[:, np.newaxis, :]
        return X_resampled, y_resampled

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, y=None
    ):
        """Make artificial samples constructed based on nearest neighbours.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_cases, n_timepoints), time series from which the new series will
            be created.

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
        X_new : ndarray
            Synthetically generated samples of shape (n_samples_new, n_timepoints).

        y_new : ndarray
            Target values for synthetic samples of shape (n_samples_new,).
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatibility with random_state
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
        X : np.ndarray
            Series from which the points will be created of shape (n_cases,
            n_timepoints).
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
        if y is not None:
            random_state = check_random_state(self.random_state)
            mask_pair_samples = y[nn_num[rows, cols]] != y_type
            diffs[mask_pair_samples] *= random_state.uniform(
                low=0.0, high=0.5, size=(mask_pair_samples.sum(), 1)
            )
        X_new = X[rows] + steps * diffs
        return X_new.astype(X.dtype)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            ClassifierChannelEnsemble provides the following special sets:
            - "results_comparison" - used in some classifiers to compare against
              previously generated results where the default set of parameters
              cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"k_neighbors": 1}
