# -*- coding: utf-8 -*-
"""Dilated Shapelet transformers.

A modification of the classic Shapelet Transform which add a dilation parameter to
Shapelets.
"""

__author__ = ["baraline"]
__all__ = ["RandomDilatedShapeletTransform"]

import warnings

import numpy as np
from numba import njit, prange, set_num_threads
from sklearn.preprocessing import LabelEncoder

from aeon.transformations.base import BaseTransformer
from aeon.utils.numba.rdst_utils import (
    choice_log,
    combinations_1d,
    get_subsequence,
    get_subsequence_with_mean_std,
    prime_up_to,
    sliding_dot_product,
    sliding_mean_std_one_series,
)
from aeon.utils.validation import check_n_jobs

__default_shapelet_lengths__ = [11]
__default_threshold_percentiles__ = [5, 10]


class RandomDilatedShapeletTransform(BaseTransformer):
    """Random Dilated Shapelet Transform (RDST) as described in [1]_[2]_.

    Overview: The input is n series with d channels of length m. First step is to
    extract candidate shapelets from the inputs. This is done randomly, and for
    each candidate shapelet:
        - Length is randomly selected from shapelet_lengths parameter
        - Dilation is sampled as a function the shapelet length and time series length
        - Normalization is choosed randomly given the probability given as parameter
        - Value is sampled randomly from an input time series given the length and
        dilation parameter.
        - Threshold is randomly choosen between two percentiles of the distribution
        of the distance vector between the shapelet and another time series, of the
        same class if they are given during fit, or to another random sample.
    Then, once the set of shapelets have been initialized, we extract the shapelet
    features from each pair of shapelets and input series. Three features are extracted:
        - min d(S,X): the minimum value of the distance vector between a shapelet S and
        a time series X.
        - argmin d(S,X): the location of the minumum.
        - SO(d(S,X), threshold): The number of point in the distance vector that are
        bellow the threshold parameter of the shapelet.

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
        the whole search space has been covered. The default is 0.5, and the maximum is
        1. Value above it have no effect for now.
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
            - mean parameter
            - standard deviation parameter

    Notes
    -----
    This implementation use all the features for multivariate shapelets, without
    affecting a random feature subsets to each shapelet as done in the original
    implementation. See `convst
    https://github.com/baraline/convst/blob/main/convst/transformers/rdst.py`_.
    It also speed up the shapelet computation with early abandoning, online
    normalization and use of the dot product to compute z-normalized squared Euclidean
    distances.

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
        shapelet_lengths=__default_shapelet_lengths__,
        proba_normalization=0.8,
        threshold_percentiles=__default_threshold_percentiles__,
        alpha_similarity=0.5,
        use_prime_dilations=False,
        random_state=None,
        n_jobs=1,
    ):
        self.max_shapelets = max_shapelets
        self.shapelet_lengths = shapelet_lengths
        self.proba_normalization = proba_normalization
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
        self._random_state = (
            np.int64(self.random_state) if isinstance(self.random_state, int) else None
        )

        self.n_instances, self.n_channels, self.series_length = X.shape

        self._check_input_params()

        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

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

        self.shapelets_ = random_dilated_shapelet_extraction(
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
        X : np.ndarray shape (n_instances, n_channels, series_length)
            The input data to transform.

        Returns
        -------
        X_new : 2D np.array of shape = (n_instances, 3*n_shapelets)
            The transformed data.
        """
        X_new = dilated_shapelet_transform(X, self.shapelets_)
        return X_new

    def _check_input_params(self):
        if isinstance(self.max_shapelets, bool):
            raise TypeError(
                "'max_shapelets' must be an integer, got {}.".format(self.max_shapelets)
            )

        if not isinstance(self.max_shapelets, (int, np.integer)):
            raise TypeError(
                "'max_shapelets' must be an integer, got {}.".format(self.max_shapelets)
            )

        if not isinstance(self.shapelet_lengths, (list, tuple, np.ndarray)):
            raise TypeError(
                "'shapelet_lengths' must be a list, a tuple or "
                "an array (got {}).".format(self.shapelet_lengths)
            )

        self.shapelet_lengths = np.array(self.shapelet_lengths, dtype=np.int64)
        if not np.all(self.shapelet_lengths >= 2):
            warnings.warn(
                "Some values in 'shapelet_lengths' are inferior to 2. These values will"
                " be ignored."
            )
            self.shapelet_lengths = self.shapelet_lengths[self.shapelet_lengths >= 2]

        if not np.all(self.shapelet_lengths <= self.series_length):
            warnings.warn(
                "All the values in 'shapelet_lengths' must be lower than or equal to "
                + "the series length. Shapelet lengths above it will be ignored."
            )
            self.shapelet_lengths = self.shapelet_lengths[
                self.shapelet_lengths <= self.series_length
            ]

        if len(self.shapelet_lengths) == 0:
            raise ValueError(
                "Shapelet lengths array is empty, did you give shapelets lengths"
                " superior to the size of the series ?"
            )

        if not isinstance(self.threshold_percentiles, (list, tuple, np.ndarray)):
            raise TypeError(
                "Expected a list, numpy array or tuple for threshold_percentiles params"
            )

        self.threshold_percentiles = np.asarray(self.threshold_percentiles)
        if len(self.threshold_percentiles) != 2:
            raise ValueError(
                "The threshold_percentiles param should be an array of size 2"
            )

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
            params = {"max_shapelets": 10}
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
    lengths = np.random.choice(shapelet_lengths, size=max_shapelets).astype(np.int64)
    # Upper bound values for dilations
    dilations = np.zeros(max_shapelets, dtype=np.int64)
    upper_bounds = np.log2(np.floor_divide(series_length - 1, lengths - 1))

    if use_prime_dilations:
        _primes = prime_up_to(np.int64(2 ** upper_bounds.max()))
        # 1 is not prime, but it is still a valid dilation for the "prime" scheme
        primes = np.zeros((_primes.shape[0] + 1), dtype=np.int64)
        primes[0] = 1
        primes[1:] = _primes
        for i in prange(max_shapelets):
            shp_primes = primes[primes <= np.int64(2 ** upper_bounds[i])]
            dilations[i] = shp_primes[choice_log(shp_primes.shape[0], 1)[0]]
    else:
        for i in prange(max_shapelets):
            dilations[i] = np.int64(2 ** np.random.uniform(0, upper_bounds[i]))

    # Init threshold array
    threshold = np.zeros(max_shapelets, dtype=np.float64)

    # Init values array
    values = np.zeros(
        (max_shapelets, n_channels, max(shapelet_lengths)), dtype=np.float64
    )

    # Is shapelet using z-normalization ?
    normalize = np.random.random(size=max_shapelets)
    normalize = normalize < proba_normalization

    means = np.zeros((max_shapelets, n_channels), dtype=np.float64)
    stds = np.zeros((max_shapelets, n_channels), dtype=np.float64)

    return values, lengths, dilations, threshold, normalize, means, stds


@njit(cache=True, parallel=True, fastmath=True)
def random_dilated_shapelet_extraction(
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
    n_instances, n_channels, series_length = X.shape
    # Fix the random seed
    if seed is not None:
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
        # (2, _, _): Mask is different for normalized and non-normalized shapelets
        alpha_mask = np.ones((2, n_instances, series_length), dtype=np.bool_)
        id_shps = np.where(dilations == unique_dil[i_dilation])[0]
        min_len = min(lengths[id_shps])
        # For each shapelet id with this dilation
        for i_shp in id_shps:
            # Get shapelet params
            dilation = dilations[i_shp]
            length = lengths[i_shp]
            norm = np.int64(normalize[i_shp])
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

    mask_values = np.ones(max_shapelets, dtype=np.bool_)
    for i in prange(max_shapelets):
        if threshold[i] == 0 and np.all(values[i] == 0):
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
def dilated_shapelet_transform(X, shapelets):
    """Perform the shapelet transform with a set of shapelets and a set of time series.

    Parameters
    ----------
    X : array, shape (n_instances, n_channels, series_length)
        Time series dataset
    Shapelets : tuple
    The returned tuple contains 7 arrays describing the shapelets parameters:
        - values : array, shape (n_shapelets, n_channels, max(shapelet_lengths))
            Values of the shapelets.
        - lengths : array, shape (n_shapelets)
            Length parameter of the shapelets
        - dilations : array, shape (n_shapelets)
            Dilation parameter of the shapelets
        - threshold : array, shape (n_shapelets)
            Threshold parameter of the shapelets
        - normalize : array, shape (n_shapelets)
            Normalization indicator of the shapelets
        - means : array, shape (n_shapelets, n_channels)
            Means of the shapelets
        - stds : array, shape (n_shapelets, n_channels)
            Standard deviation of the shapelets

    Returns
    -------
    X_new : array, shape=(n_instances, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.

    """
    (values, lengths, dilations, threshold, normalize, means, stds) = shapelets
    n_shapelets = len(lengths)
    n_instances, n_channels, series_length = X.shape

    n_ft = 3

    # (u_l * u_d , 2)
    params_shp = combinations_1d(lengths, dilations)

    X_new = np.zeros((n_instances, n_ft * n_shapelets))
    for i_params in prange(params_shp.shape[0]):
        length = params_shp[i_params, 0]
        dilation = params_shp[i_params, 1]
        id_shps = np.where((lengths == length) & (dilations == dilation))[0]
        for i_x in prange(n_instances):
            idx_no_norm = id_shps[np.where(~normalize[id_shps])[0]]
            for i_shp in idx_no_norm:
                X_new[
                    i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)
                ] = compute_shapelet_features(
                    X[i_x], values[i_shp], length, dilation, threshold[i_shp]
                )

            idx_norm = id_shps[np.where(normalize[id_shps])[0]]
            if len(idx_norm) > 0:
                X_means, X_stds = sliding_mean_std_one_series(X[i_x], length, dilation)
                for i_shp in idx_norm:
                    X_new[
                        i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)
                    ] = compute_shapelet_features_normalized(
                        X[i_x],
                        values[i_shp],
                        length,
                        dilation,
                        threshold[i_shp],
                        X_means,
                        X_stds,
                        means[i_shp],
                        stds[i_shp],
                    )
    return X_new


@njit(cache=True, fastmath=True)
def compute_shapelet_features(X, values, length, dilation, threshold):
    """Extract the features from a shapelet distance vector.

    Given a shapelet and a time series, extract three features from the resulting
    distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X : array, shape (n_channels, series_length)
        An input time series
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet
    values : array, shape (n_channels, length)
        The resulting subsequence
    X_means : array, shape (n_channels)
        The mean of each subsequence (l,d) of channel of the input time series
    X_stds: array, shape (n_channels)
        The standard deviation of each subsequence (l,d) of channel of the
        input time series
    means : array, shape (n_channels)
        The mean of each channel of the shapelet
    stds: array, shape (n_channels)
        The std of each channel of the shapelet

    Returns
    -------
    min, argmin, shapelet occurence
        The three computed features as float dtypes
    """
    _min = np.inf
    _argmin = np.inf
    _SO = 0

    n_channels, series_length = X.shape
    n_subs = series_length - (length - 1) * dilation
    for i_sub in prange(n_subs):
        idx = i_sub
        _sum = 0
        for i_l in prange(length):
            for i_channel in prange(n_channels):
                _sum += (X[i_channel, idx] - values[i_channel, i_l]) ** 2

            if _sum >= _min and _sum >= threshold:
                break

            idx += dilation

        if _sum < _min:
            _min = _sum
            _argmin = i_sub
        if _sum < threshold:
            _SO += 1

    return _min, np.float64(_argmin), np.float64(_SO)


@njit(cache=True, fastmath=True)
def compute_shapelet_features_normalized(
    X, values, length, dilation, threshold, X_means, X_stds, means, stds
):
    """Extract the features from a normalized shapelet distance vector.

    Given a shapelet, a time series and their means and standard deviations, extract
    three features from the resulting normalized distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X : array, shape (n_channels, series_length)
        An input time series
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet
    values : array, shape (n_channels, length)
        The resulting subsequence
    X_means : array, shape (n_channels)
        The mean of each subsequence (l,d) of channel of the input time series
    X_stds: array, shape (n_channels)
        The standard deviation of each subsequence (l,d) of channel of the
        input time series
    means : array, shape (n_channels)
        The mean of each channel of the shapelet
    stds: array, shape (n_channels)
        The std of each channel of the shapelet

    Returns
    -------
    min, argmin, shapelet occurence
        The three computed features as float dtypes

    """
    _min = np.inf
    _argmin = np.inf
    _SO = 0
    n_channels, series_length = X.shape
    n_subs = series_length - (length - 1) * dilation
    for i_sub in prange(n_subs):
        _sum = 0
        for i_channel in prange(n_channels):
            idx = i_sub
            dot_sub = 0
            for i_l in prange(length):
                dot_sub += X[i_channel, idx] * values[i_channel, i_l]
                idx += dilation

            if stds[i_channel] <= 0:
                _sum += X_stds[i_channel, i_sub] * length
            else:
                if X_stds[i_channel, i_sub] <= 0:
                    _sum += stds[i_channel] * length
                else:
                    denom = length * stds[i_channel] * X_stds[i_channel, i_sub]
                    p = (
                        dot_sub - length * means[i_channel] * X_means[i_channel, i_sub]
                    ) / denom
                    p = min(p, 1.0)
                    _sum += abs(2 * length * (1.0 - p))

            if _sum >= _min and _sum >= threshold:
                break

        if _sum < _min:
            _min = _sum
            _argmin = i_sub
        if _sum < threshold:
            _SO += 1

    return _min, np.float64(_argmin), np.float64(_SO)


@njit(cache=True, fastmath=True)
def compute_normalized_shapelet_dist_vector(X, values, length, dilation, means, stds):
    """Compute the normalized distance vector between a shapelet and a time series.

    Parameters
    ----------
    X : array, shape (n_channels, series_length)
        An input time series
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet
    means : array, shape (n_channels)
        The mean of each channel of the shapelet
    stds: array, shape (n_channels)
        The std of each channel of the shapelet

    Returns
    -------
    d_vect : array, shape (series_length - (length-1) * dilation)
        The resulting distance vector
    """
    n_channels, series_length = X.shape
    # shape (n_channels, n_subsequences)
    X_means, X_stds = sliding_mean_std_one_series(X, length, dilation)
    X_dots = sliding_dot_product(X, values, length, dilation)

    d_vect_len = series_length - (length - 1) * dilation
    d_vect = np.zeros(d_vect_len)
    for i_channel in prange(n_channels):
        # Edge case: shapelet channel is constant
        if stds[i_channel] <= 0:
            for i_sub in prange(d_vect_len):
                d_vect[i_sub] += X_stds[i_channel, i_sub] * length
        else:
            for i_sub in prange(d_vect_len):
                # Edge case: subsequence channel is constant
                if X_stds[i_channel, i_sub] <= 0:
                    d_vect[i_sub] += stds[i_channel] * length
                else:
                    denom = length * stds[i_channel] * X_stds[i_channel, i_sub]
                    p = (
                        X_dots[i_channel, i_sub]
                        - length * means[i_channel] * X_means[i_channel, i_sub]
                    ) / denom
                    p = min(p, 1.0)
                    d_vect[i_sub] += abs(2 * length * (1.0 - p))
    return d_vect


@njit(cache=True, fastmath=True)
def compute_shapelet_dist_vector(X, values, length, dilation):
    """Compute the distance vector between a shapelet and a time series.

    Parameters
    ----------
    X : array, shape (n_channels, series_length)
        An input time series
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet

    Returns
    -------
    d_vect : array, shape (series_length - (length-1) * dilation)
        The resulting distance vector
    """
    n_channels, series_length = X.shape
    d_vect_len = series_length - (length - 1) * dilation
    d_vect = np.zeros(d_vect_len)
    for i_vect in prange(d_vect_len):
        for i_channel in prange(n_channels):
            _idx = i_vect
            for i_l in prange(length):
                d_vect[i_vect] += (X[i_channel, _idx] - values[i_channel, i_l]) ** 2
                _idx += dilation
    return d_vect
