# -*- coding: utf-8 -*-
"""Dilated Shapelet transformers.

A modification of the classic Shapelet Transform which add a dilation parameter to
Shapelets.
"""

__author__ = ["baraline"]
__all__ = ["RandomDilatedShapeletTransform"]

import numpy as np
from numba import njit, prange, set_num_threads
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from aeon.transformations.base import BaseTransformer
from aeon.utils.numba.rdst_utils import (
    choice_log,
    compute_normalized_shapelet_dist_vector,
    compute_shapelet_dist_vector,
    get_subsequence,
    get_subsequence_with_mean_std,
    prime_up_to,
)
from aeon.utils.validation import check_n_jobs

# todo: if any imports are aeon soft dependencies:
#  * make sure to fill in the "python_dependencies" tag with the package import name
#  * add a _check_soft_dependencies warning here, example:
#
# from aeon.utils.validation._dependencies import check_soft_dependencies
# _check_soft_dependencies("soft_dependency_name", severity="warning")


class RandomDilatedShapeletTransform(BaseTransformer):
    """Random Dilated Shapelet Transform (RDST) as described in [1]_[2]_.

    Overview: Input n series with d channels of length m. First step is to extract
    candidate shapelets from the inputs.
        For each candidate shapelet:
            - TODO
    Then, once the set of shapelets have been initialized, we then extract the shapelet
    features from each pair of shapelets and input series.
        We extract three features in
            - TODO describe computations

    Parameters
    ----------
    max_shapelets : int, default=10000
        The maximum number of shapelet to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array, default=[11]
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set.
    proba_normalization : float, default=0.8
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    threshold_percentiles : array, default=[5,10]
        The two perceniles used to select the threshold used to compute the Shapelet
        Occurrence feature.
    alpha_similarity : float, default=0.5
        The strenght of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5.
    use_prime_dilations : bool, default=False
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed-up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    n_jobs : int, default=1
        The number of threads used for both `fit` and `transform`.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    shapelets : list
        The stored shapelets. Each item in the list is a tuple containing:
            - shapelet values
            - length parameter
            - dilation parameter
            - treshold parameter
            - normalization parameter

    Notes
    -----
    This implementation use all the features for multivariate shapelets, without
    affecting a random feature subsets to each shapelet as done in the original
    implementation. See `convst
    https://github.com/baraline/convst/blob/main/convst/transformers/rdst.py`_.

    References
    ----------
    .. [1] Antoine Guillaume et al. "Random Dilated Shapelet Transform: A New Approach
       for Time Series Shapelets", Pattern Recognition and Artificial Intelligence.
       ICPRAI 2022.
    .. [2] Antoine Guillaume, "Time series classification with shapelets: Application
       to predictive maintenance on event logs", PhD Thesis, University of Orléans,
       2023.

    Examples
    --------
    >>> from aeon.transformations.panel.dilated_shapelet_transform import (
    ...     RandomDilatedShapeletTransform
    ... )
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> t = RandomDilatedShapeletTransform(
    ...     max_shapelets=10
    ... )
    >>> t.fit(X_train, y_train)
    RandomDilatedShapeletTransform(...)
    >>> X_t = t.transform(X_train)
    """

    # Variable length is defined as variable length for fit transform(but same in one X)
    # or variable within one X
    # How do we specify y not mandatory but accepted
    _tags = {
        "scitype:transform-output": "Primitives",
        "fit_is_empty": False,
        "univariate-only": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "requires_y": False,
        "capability:inverse_transform": False,
        "handles-missing-data": False,
    }

    def __init__(
        self,
        max_shapelets=10_000,
        shapelet_lengths=None,
        proba_normalization=0.8,
        threshold_percentiles=None,
        alpha_similarity=0.5,
        use_prime_dilations=False,
        random_state=None,
        n_jobs=1,
    ):
        self.max_shapelets = max_shapelets
        if shapelet_lengths is None:
            self.shapelet_lengths = [11]
        else:
            self.shapelet_lengths = shapelet_lengths
        self.proba_normalization = proba_normalization
        if threshold_percentiles is None:
            self.threshold_percentiles = [5, 10]
        else:
            self.threshold_percentiles = threshold_percentiles
        self.alpha_similarity = alpha_similarity
        self.use_prime_dilations = use_prime_dilations
        self.random_state = random_state
        self.n_jobs = n_jobs

        super(RandomDilatedShapeletTransform, self).__init__()

    def _fit(self, X, y=None):
        """Fit the random dilated shapelet transform to a specified X and y.

        Parameters
        ----------
        X: np.ndarray shape (n_instances, n_channels, series_length)
            The training input samples.
        y: array-like or list, default=None
            The class values for X. If not specified, a random sample (i.e. not of the
            same class) will be used when computing the threshold for the Shapelet
            Occurence feature.

        Returns
        -------
        self : RandomDilatedShapeletTransform
            This estimator.
        """
        # Does forcing float32 as in MiniRocket improves performance ?
        self._random_state = check_random_state(self.random_state)

        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

        self.n_instances, self.n_channels, self.series_length = X.shape

        if y is None:
            y = np.zeros(self.n_instances)
        else:
            y = LabelEncoder().fit_transform(y)

        if any(self.shapelet_lengths > self.series_length):
            raise ValueError(
                "Shapelets lengths can't be superior to input length,",
                "but got shapelets_lengths = {} ".format(self.shapelet_lengths),
                "with input length = {}".format(self.series_length),
            )

        self.shapelets_ = _random_dilated_shapelet_extraction(
            X,
            y,
            self.max_shapelets,
            self.shapelet_lengths,
            self.proba_normalization,
            self.threshold_percentiles,
            self.alpha_similarity,
            self.use_prime_dilations,
            self._random_state,
        )

        return self

    def _transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : np.ndarray shape (n_time_series, n_channels, series_length)
            The input data to transform.

        Returns
        -------
        X_new : 2D np.array of shape = (n_instances, 3*max_shapelets)
            The transformed data.
        """
        X_new = _dilated_shapelet_transform(X, self.shapelets_)
        return X_new

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        if parameter_set == "default":
            params = {"max_shapelets": 5, "n_shapelet_samples": 50, "batch_size": 20}
        else:
            raise NotImplementedError(
                "The parameter set {} is not yet implemented".format(parameter_set)
            )
        return params


@njit(cache=True, fastmath=True)
def _init_random_shapelet_params(
    max_shapelets,
    shapelet_lengths,
    proba_normalization,
    use_prime_dilations,
    n_channels,
    series_length,
):
    """Randomly initialize the parameters of the shapelets.

    Parameters
    ----------
    max_shapelets : int
        The maximum number of shapelet to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set.
    proba_normalization : float
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    use_prime_dilations : bool
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed-up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    n_channels : int
        Number of channels of the input time series.
    series_length : int
        Size of the input time series.

    Returns
    -------
    values : array, shape (max_shapelets, n_channels, max(shapelet_lengths))
        An initialized (empty) value array for each shapelet
    lengths : array, shape (max_shapelets)
        The randomly initialized length of each shapelet
    dilations : array, shape (max_shapelets)
        The randomly initialized dilation of each shapelet
    threshold : array, shape (max_shapelets)
        An initialized (empty) value array for each shapelet
    normalize : array, shape (max_shapelets)
        The randomly initialized normalization indicator of each shapelet
    - means : array, shape (max_shapelets, n_channels)
        Means of the shapelets
    - stds : array, shape (max_shapelets, n_channels)
        Standard deviation of the shapelets

    """
    # Lengths of the shapelets
    # test dtypes correctness
    lengths = np.random.choice(shapelet_lengths, size=max_shapelets)
    # Upper bound values for dilations
    dilations = np.zeros(max_shapelets, dtype=int)
    upper_bounds = np.log2(np.floor_divide(series_length - 1, lengths - 1))

    if use_prime_dilations:
        _primes = prime_up_to(int(2 ** upper_bounds.max()))
        # 0 and 1 are not primes, but we need them for 2**0 =1 and 2**1 = 2
        primes = np.zeros(_primes.shape[0] + 2)
        primes[1] = 1
        primes[2:] = _primes
        for i in prange(max_shapelets):
            shp_primes = primes[primes <= int(2 ** upper_bounds[i])]
            dilations[i] = shp_primes[choice_log(shp_primes.shape[0], 1)[0]]
    else:
        for i in prange(max_shapelets):
            dilations[i] = int(2 ** np.random.uniform(0, upper_bounds[i]))

    # Init threshold array
    threshold = np.zeros(max_shapelets)

    # Init values array
    values = np.zeros((max_shapelets, n_channels, max(shapelet_lengths)))

    # Is shapelet using z-normalization ?
    normalize = np.random.random(size=max_shapelets)
    normalize = normalize < proba_normalization

    means = np.zeros((max_shapelets, n_channels))
    stds = np.zeros((max_shapelets, n_channels))

    return values, lengths, dilations, threshold, normalize, means, stds


@njit(cache=True, parallel=True, fastmath=True)
def _random_dilated_shapelet_extraction(
    X,
    y,
    max_shapelets,
    shapelet_lengths,
    proba_normalization,
    threshold_percentiles,
    alpha_similarity,
    use_prime_dilations,
    seed,
):
    """Randomly generate a set of shapelets given the input parameters.

    Parameters
    ----------
    X : array, shape (n_instances, n_channels, series_length)
        Time series dataset
    y : array, shape (n_instances)
        Class of each input time series
    max_shapelets : int
        The maximum number of shapelet to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set.
    proba_normalization : float
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    threshold_percentiles : array
        The two perceniles used to select the threshold used to compute the Shapelet
        Occurrence feature.
    alpha_similarity : float
        The strenght of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5.
    use_prime_dilations : bool
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed-up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    random_state : int
        Seed for random number generation.

    Returns
    -------
    Shapelets : tuple
    The returned tuple contains 7 arrays describing the shapelets parameters:
        - values : array, shape (max_shapelets, n_channels, max(shapelet_lengths))
            Values of the shapelets.
        - lengths : array, shape (max_shapelets)
            Length parameter of the shapelets
        - dilations : array, shape (max_shapelets)
            Dilation parameter of the shapelets
        - threshold : array, shape (max_shapelets)
            Threshold parameter of the shapelets
        - normalize : array, shape (max_shapelets)
            Normalization indicator of the shapelets
        - means : array, shape (max_shapelets, n_channels)
            Means of the shapelets
        - stds : array, shape (max_shapelets, n_channels)
            Standard deviation of the shapelets
    """
    # TODO : using python dtype for now, check with test if that adapts to any dtype
    # TODO : check if going all dtype 32 improve performances (question slack)
    n_instances, n_channels, series_length = X.shape
    # Fix the random seed
    np.random.seed(seed)

    # Initialize shapelets
    (
        values,
        lengths,
        dilations,
        threshold,
        normalize,
        means,
        stds,
    ) = _init_random_shapelet_params(
        max_shapelets,
        shapelet_lengths,
        proba_normalization,
        use_prime_dilations,
        n_channels,
        series_length,
    )
    # Get unique dilations to loop over
    unique_dil = np.unique(dilations)
    n_dilations = unique_dil.shape[0]

    # For each dilation, we can do in parallel
    for i_dilation in prange(n_dilations):
        alpha_mask = np.ones((2, n_instances, series_length), dtype=bool)
        id_shps = np.where(dilations == unique_dil[i_dilation])[0]
        min_len = min(lengths[id_shps])
        # For each shapelet id with this dilation
        for i_shp in id_shps:
            # Get shapelet params
            dilation = dilations[i_shp]
            length = lengths[i_shp]
            norm = int(normalize[i_shp])
            dist_vect_shape = series_length - (length - 1) * dilation

            # Possible sampling points given self similarity mask
            current_mask = alpha_mask[norm, :, :dist_vect_shape]
            idx_mask = np.where(current_mask)

            n_admissible_points = idx_mask[0].shape[0]
            if n_admissible_points > 0:
                # Choose a sample and a timestamp
                idx_choice = np.random.choice(n_admissible_points)
                idx_sample = idx_mask[0][idx_choice]
                idx_timestamp = idx_mask[1][idx_choice]

                # Update the mask in two directions from the sampling point
                alpha_size = length - int(max(1, (1 - alpha_similarity) * min_len))
                for j in range(alpha_size):
                    alpha_mask[
                        norm, idx_sample, (idx_timestamp - (j * dilation))
                    ] = False
                    alpha_mask[
                        norm, idx_sample, (idx_timestamp + (j * dilation))
                    ] = False

                # Extract the values of shapelet
                if norm:
                    _val, _means, _stds = get_subsequence_with_mean_std(
                        X[idx_sample], idx_timestamp, length, dilation
                    )
                else:
                    _val = get_subsequence(
                        X[idx_sample], idx_timestamp, length, dilation
                    )

                # Select another sample of the same class as the sample used to
                loc_others = np.where(y == y[idx_sample])[0]
                if loc_others.shape[0] > 1:
                    loc_others = loc_others[loc_others != idx_sample]
                    id_test = np.random.choice(loc_others)
                else:
                    id_test = idx_sample

                # Compute distance vector
                if norm:
                    x_dist = compute_normalized_shapelet_dist_vector(
                        X[id_test], _val, length, dilation, _means, _stds
                    )
                else:
                    x_dist = compute_shapelet_dist_vector(
                        X[id_test],
                        _val,
                        length,
                        dilation,
                    )

                lower_bound = np.percentile(x_dist, threshold_percentiles[0])
                upper_bound = np.percentile(x_dist, threshold_percentiles[0])

                threshold[i_shp] = np.random.uniform(lower_bound, upper_bound)
                values[i_shp, :, :length] = _val
                if norm:
                    means[i_shp] = _means
                    stds[i_shp] = _stds

    mask_values = np.ones(max_shapelets, dtype=bool)
    for i in prange(max_shapelets):
        if np.all(values[i] == 0):
            mask_values[i] = False

    return (
        values[mask_values],
        lengths[mask_values],
        dilations[mask_values],
        threshold[mask_values],
        normalize[mask_values],
        means[mask_values],
        stds[mask_values],
    )


@njit(cache=True, parallel=True, fastmath=True)
def _dilated_shapelet_transform(X, shapelets):
    # TODO : To implement
    return X
