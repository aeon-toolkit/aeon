"""Dilated Shapelet transformers.

A modification of the classic Shapelet Transform which add a dilation parameter to
Shapelets.
"""

__maintainer__ = []
__all__ = ["RandomDilatedShapeletTransform"]

import warnings

import numpy as np
from numba import njit, prange, set_num_threads
from sklearn.preprocessing import LabelEncoder

from aeon.distances import manhattan_distance
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    choice_log,
    combinations_1d,
    get_subsequence,
    get_subsequence_with_mean_std,
    set_numba_random_seed,
    sliding_mean_std_one_series,
)
from aeon.utils.numba.stats import prime_up_to
from aeon.utils.validation import check_n_jobs


class RandomDilatedShapeletTransform(BaseCollectionTransformer):
    """Random Dilated Shapelet Transform (RDST) as described in [1]_[2]_.

    Overview: The input is n series with d channels of length m. First step is to
    extract candidate shapelets from the inputs. This is done randomly, and for
    each candidate shapelet:
        - Length is randomly selected from shapelet_lengths parameter
        - Dilation is sampled as a function the shapelet length and time series length
        - Normalization is chosen randomly given the probability given as parameter
        - Value is sampled randomly from an input time series given the length and
        dilation parameter.
        - Threshold is randomly chosen between two percentiles of the distribution
        of the distance vector between the shapelet and another time series. This time
        serie is drawn from the same class if classes are given during fit. Otherwise,
        a random sample will be used. If there is only one sample per class, the same
        sample will be used.
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
    shapelet_lengths : array, default=None
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set. If None, the shapelets length will be equal to
        min(max(2,n_timepoints//2),11).
    proba_normalization : float, default=0.8
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    threshold_percentiles : array, default=None
        The two percentiles used to select the threshold used to compute the Shapelet
        Occurrence feature. If None, the 5th and the 10th percentiles (i.e. [5,10])
        will be used.
    alpha_similarity : float, default=0.5
        The strength of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5, and the maximum is
        1. Value above it have no effect for now.
    use_prime_dilations : bool, default=False
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed up the algorithm for long time series and/or
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
    max_shapelet_length_ : int
        The maximum actual shapelet length fitted to train data.
    min_n_timepoints_ : int
        The minimum length of series in train data.

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
    >>> from aeon.transformations.collection.shapelet_based import (
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
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "y_inner_type": "numpy1D",
        "algorithm_type": "shapelet",
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
        self.shapelet_lengths = shapelet_lengths
        self.proba_normalization = proba_normalization
        self.threshold_percentiles = threshold_percentiles
        self.alpha_similarity = alpha_similarity
        self.use_prime_dilations = use_prime_dilations
        self.random_state = random_state
        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, X, y=None):
        """Fit the random dilated shapelet transform to a specified X and y.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
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
        # Numba does not yet support new random numpy API with generator
        if isinstance(self.random_state, int):
            self._random_state = np.int32(self.random_state)
        else:
            self._random_state = np.int32(np.random.randint(0, 2**31))

        n_cases_ = len(X)
        self.min_n_timepoints_ = min([X[i].shape[1] for i in range(n_cases_)])

        self._check_input_params()

        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

        if y is None:
            y = np.zeros(n_cases_)
        else:
            y = LabelEncoder().fit_transform(y)

        if any(self.shapelet_lengths_ > self.min_n_timepoints_):
            raise ValueError(
                "Shapelets lengths can't be superior to input length,",
                f"but got shapelets_lengths = {self.shapelet_lengths_} ",
                f"with an input length = {self.min_n_timepoints_}",
            )
        self.shapelets_ = random_dilated_shapelet_extraction(
            X,
            y,
            self.max_shapelets,
            self.shapelet_lengths_,
            self.proba_normalization,
            self.threshold_percentiles_,
            self.alpha_similarity,
            self.use_prime_dilations,
            self._random_state,
        )
        if len(self.shapelets_[0]) == 0:
            raise RuntimeError(
                "No shapelets were extracted during the fit method with the specified"
                " parameters."
            )
        if np.isnan(self.shapelets_[0]).any():
            raise RuntimeError(
                "Got NaN values in the extracted shapelet values. This may happen if "
                "you have NaN values in your data. We do not currently support NaN "
                "values for shapelet transformation."
            )

        # Shapelet "length" is length-1 times dilation
        self.max_shapelet_length_ = np.max(
            (self.shapelets_[1] - 1) * self.shapelets_[2]
        )

        return self

    def _transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            The input data to transform.

        Returns
        -------
        X_new : 2D np.array of shape = (n_cases, 3*n_shapelets)
            The transformed data.
        """
        for i in range(0, len(X)):
            if X[i].shape[1] < self.max_shapelet_length_:
                raise ValueError(
                    "The shortest series in transform is smaller than "
                    "the min shapelet length, pad to min length prior to "
                    "calling transform."
                )

        X_new = dilated_shapelet_transform(X, self.shapelets_)
        if np.isinf(X_new).any() or np.isnan(X_new).any():
            warnings.warn(
                "Some invalid values (inf or nan) where converted from to 0 during the"
                " shapelet transformation.",
                stacklevel=2,
            )
            X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)

        return X_new

    def _check_input_params(self):
        if isinstance(self.max_shapelets, bool):
            raise TypeError(
                f"'max_shapelets' must be an integer, got {self.max_shapelets}."
            )

        if not isinstance(self.max_shapelets, (int, np.integer)):
            raise TypeError(
                f"'max_shapelets' must be an integer, got {self.max_shapelets}."
            )
        self.shapelet_lengths_ = self.shapelet_lengths
        if self.shapelet_lengths_ is None:
            self.shapelet_lengths_ = np.array(
                [min(max(2, self.min_n_timepoints_ // 2), 11)]
            )
        else:
            if not isinstance(self.shapelet_lengths_, (list, tuple, np.ndarray)):
                raise TypeError(
                    "'shapelet_lengths' must be a list, a tuple or "
                    "an array (got {}).".format(self.shapelet_lengths_)
                )

            self.shapelet_lengths_ = np.array(self.shapelet_lengths_, dtype=np.int32)
            if not np.all(self.shapelet_lengths_ >= 2):
                warnings.warn(
                    "Some values in 'shapelet_lengths' are inferior to 2."
                    "These values will be ignored.",
                    stacklevel=2,
                )
                self.shapelet_lengths_ = self.shapelet_lengths[
                    self.shapelet_lengths_ >= 2
                ]

            if not np.all(self.shapelet_lengths_ <= self.min_n_timepoints_):
                warnings.warn(
                    "All the values in 'shapelet_lengths' must be lower or equal to"
                    + "the series length. Shapelet lengths above it will be ignored.",
                    stacklevel=2,
                )
                self.shapelet_lengths_ = self.shapelet_lengths_[
                    self.shapelet_lengths_ <= self.min_n_timepoints_
                ]

            if len(self.shapelet_lengths_) == 0:
                raise ValueError(
                    "Shapelet lengths array is empty, did you give shapelets lengths"
                    " superior to the size of the series ?"
                )

        self.threshold_percentiles_ = self.threshold_percentiles
        if self.threshold_percentiles_ is None:
            self.threshold_percentiles_ = np.array([5, 10])
        else:
            if not isinstance(self.threshold_percentiles_, (list, tuple, np.ndarray)):
                raise TypeError(
                    "Expected a list, numpy array or tuple for threshold_percentiles"
                )
            if len(self.threshold_percentiles_) != 2:
                raise ValueError(
                    "The threshold_percentiles param should be an array of size 2"
                )
            self.threshold_percentiles_ = np.asarray(self.threshold_percentiles_)

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
                f"The parameter set {parameter_set} is not yet implemented"
            )
        return params


@njit(fastmath=True, cache=True)
def _init_random_shapelet_params(
    max_shapelets,
    shapelet_lengths,
    proba_normalization,
    use_prime_dilations,
    n_channels,
    n_timepoints,
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
        values. This can greatly speed up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    n_channels : int
        Number of channels of the input time series.
    n_timepoints : int
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
    means : array, shape (max_shapelets, n_channels)
        Means of the shapelets
    stds : array, shape (max_shapelets, n_channels)
        Standard deviation of the shapelets

    """
    # Lengths of the shapelets
    # test dtypes correctness
    lengths = np.random.choice(shapelet_lengths, size=max_shapelets).astype(np.int32)
    # Upper bound values for dilations
    dilations = np.zeros(max_shapelets, dtype=np.int32)
    upper_bounds = np.log2(np.floor_divide(n_timepoints - 1, lengths - 1))

    if use_prime_dilations:
        _primes = prime_up_to(np.int32(2 ** upper_bounds.max()))
        # 1 is not prime, but it is still a valid dilation for the "prime" scheme
        primes = np.zeros((_primes.shape[0] + 1), dtype=np.int32)
        primes[0] = 1
        primes[1:] = _primes
        for i in prange(max_shapelets):
            shp_primes = primes[primes <= np.int32(2 ** upper_bounds[i])]
            dilations[i] = shp_primes[choice_log(shp_primes.shape[0], 1)[0]]
    else:
        for i in prange(max_shapelets):
            dilations[i] = np.int32(2 ** np.random.uniform(0, upper_bounds[i]))

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


@njit(cache=True)
def _get_admissible_sampling_point(current_mask):
    n_cases = len(current_mask)
    # Count the number of admissible points per sample as cumsum
    n_admissible_points = 0
    for i in range(n_cases):
        n_admissible_points += current_mask[i].shape[0]
    if n_admissible_points > 0:
        idx_choice = np.random.choice(n_admissible_points)
        for i in range(n_cases):
            _new_val = idx_choice - current_mask[i].shape[0]
            if _new_val < 0 and current_mask[i].shape[0] > 0:
                return i, idx_choice
            idx_choice = _new_val
    else:
        return -1, -1


@njit(fastmath=True, cache=True, parallel=True)
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
    X : array, shape (n_cases, n_channels, n_timepoints)
        Time series dataset
    y : array, shape (n_cases)
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
        The two percentiles used to select the threshold used to compute the Shapelet
        Occurrence feature.
    alpha_similarity : float
        The strength of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5.
    use_prime_dilations : bool
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    seed : int
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
    n_cases = len(X)
    n_channels = X[0].shape[0]
    n_timepointss = np.zeros(n_cases, dtype=np.int64)
    for i in range(n_cases):
        n_timepointss[i] = X[i].shape[1]
    min_n_timepoints = n_timepointss.min()
    max_n_timepoints = n_timepointss.max()
    # Fix the random seed
    set_numba_random_seed(seed)

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
        min_n_timepoints,
    )
    # Get unique dilations to loop over
    unique_dil = np.unique(dilations)
    n_dilations = unique_dil.shape[0]

    # For each dilation, we can do in parallel
    for i_dilation in prange(n_dilations):
        # (2, _, _): Mask is different for normalized and non-normalized shapelets
        alpha_mask = np.ones((2, n_cases, max_n_timepoints), dtype=np.bool_)
        for _i in range(n_cases):
            # For the unequal length case, we scale the mask up and set to False
            alpha_mask[:, _i, n_timepointss[_i] :] = False

        id_shps = np.where(dilations == unique_dil[i_dilation])[0]
        min_len = min(lengths[id_shps])
        # For each shapelet id with this dilation
        for i_shp in id_shps:
            # Get shapelet params
            dilation = dilations[i_shp]
            length = lengths[i_shp]
            norm = np.int_(normalize[i_shp])
            # Possible sampling points given self similarity mask
            current_mask = [
                np.where(
                    alpha_mask[norm, _i, : n_timepointss[_i] - (length - 1) * dilation]
                )[0]
                for _i in range(n_cases)
            ]
            idx_sample, idx_timestamp = _get_admissible_sampling_point(current_mask)
            if idx_sample >= 0:
                # Update the mask in two directions from the sampling point
                alpha_size = length - int(max(1, (1 - alpha_similarity) * min_len))
                for j in range(alpha_size):
                    alpha_mask[norm, idx_sample, (idx_timestamp - (j * dilation))] = (
                        False
                    )
                    alpha_mask[norm, idx_sample, (idx_timestamp + (j * dilation))] = (
                        False
                    )

                # Extract the values of shapelet
                if norm:
                    _val, _means, _stds = get_subsequence_with_mean_std(
                        X[idx_sample], idx_timestamp, length, dilation
                    )
                    for i_channel in prange(_val.shape[0]):
                        if _stds[i_channel] > AEON_NUMBA_STD_THRESHOLD:
                            _val[i_channel] = (
                                _val[i_channel] - _means[i_channel]
                            ) / _stds[i_channel]
                        else:
                            _val[i_channel] = _val[i_channel] - _means[i_channel]
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

                # Compute distance vector, first get the subsequences
                X_subs = get_all_subsequences(X[id_test], length, dilation)
                if norm:
                    # Normalize them if needed
                    X_means, X_stds = sliding_mean_std_one_series(
                        X[id_test], length, dilation
                    )
                    X_subs = normalize_subsequences(X_subs, X_means, X_stds)
                x_dist = compute_shapelet_dist_vector(X_subs, _val, length)

                lower_bound = np.percentile(x_dist, threshold_percentiles[0])
                upper_bound = np.percentile(x_dist, threshold_percentiles[1])

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


@njit(fastmath=True, cache=True, parallel=True)
def dilated_shapelet_transform(X, shapelets):
    """Perform the shapelet transform with a set of shapelets and a set of time series.

    Parameters
    ----------
    X : array, shape (n_cases, n_channels, n_timepoints)
        Time series dataset
    shapelets : tuple
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
    X_new : array, shape=(n_cases, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.

    """
    (values, lengths, dilations, threshold, normalize, means, stds) = shapelets
    n_shapelets = len(lengths)
    n_cases = len(X)
    n_ft = 3

    # (u_l * u_d , 2)
    params_shp = combinations_1d(lengths, dilations)

    X_new = np.zeros((n_cases, n_ft * n_shapelets))
    for i_params in prange(params_shp.shape[0]):
        length = params_shp[i_params, 0]
        dilation = params_shp[i_params, 1]
        id_shps = np.where((lengths == length) & (dilations == dilation))[0]

        for i_x in prange(n_cases):
            X_subs = get_all_subsequences(X[i_x], length, dilation)
            idx_no_norm = id_shps[np.where(~normalize[id_shps])[0]]
            for i_shp in idx_no_norm:
                X_new[i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)] = (
                    compute_shapelet_features(
                        X_subs, values[i_shp], length, threshold[i_shp]
                    )
                )

            idx_norm = id_shps[np.where(normalize[id_shps])[0]]
            if len(idx_norm) > 0:
                X_means, X_stds = sliding_mean_std_one_series(X[i_x], length, dilation)
                X_subs = normalize_subsequences(X_subs, X_means, X_stds)
                for i_shp in idx_norm:
                    X_new[i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)] = (
                        compute_shapelet_features(
                            X_subs, values[i_shp], length, threshold[i_shp]
                        )
                    )
    return X_new


@njit(fastmath=True, cache=True)
def normalize_subsequences(X_subs, X_means, X_stds):
    """
    Generate subsequences from a time series given the length and dilation parameters.

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    X_means : array, shape (n_channels, n_timestamps-(length-1)*dilation)
        Length of the subsequences to generate.
    X_stds : array, shape (n_channels, n_timestamps-(length-1)*dilation)
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_timestamps-(length-1)*dilation, n_channels, length)
        Subsequences of the input time series.
    """
    n_subsequences, n_channels, length = X_subs.shape
    X_new = np.zeros((n_subsequences, n_channels, length))
    for i_sub in prange(n_subsequences):
        for i_channel in prange(n_channels):
            if X_stds[i_channel, i_sub] > AEON_NUMBA_STD_THRESHOLD:
                X_new[i_sub, i_channel] = (
                    X_subs[i_sub, i_channel] - X_means[i_channel, i_sub]
                ) / X_stds[i_channel, i_sub]
            # else it gives 0, the default value
    return X_new


@njit(fastmath=True, cache=True)
def get_all_subsequences(X, length, dilation):
    """
    Generate subsequences from a time series given the length and dilation parameters.

    Parameters
    ----------
    X : array, shape = (n_channels, n_timestamps)
        An input time series as (n_channels, n_timestamps).
    length : int
        Length of the subsequences to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_timestamps-(length-1)*dilation, n_channels, length)
        Subsequences of the input time series.
    """
    n_channels, n_timestamps = X.shape
    n_subsequences = n_timestamps - (length - 1) * dilation
    X_subs = np.zeros((n_subsequences, n_channels, length))
    for i_sub in prange(n_subsequences):
        for i_channel in prange(n_channels):
            for i_length in prange(length):
                X_subs[i_sub, i_channel, i_length] = X[
                    i_channel, i_sub + (i_length * dilation)
                ]
    return X_subs


@njit(fastmath=True, cache=True)
def compute_shapelet_features(X_subs, values, length, threshold):
    """Extract the features from a shapelet distance vector.

    Given a shapelet and a time series, extract three features from the resulting
    distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    values : array, shape (n_channels, length)
        The resulting subsequence
    threshold : float
        The threshold parameter of the shapelet

    Returns
    -------
    min, argmin, shapelet occurence
        The three computed features as float dtypes
    """
    _min = np.inf
    _argmin = np.inf
    _SO = 0

    n_subsequences = X_subs.shape[0]

    for i_sub in prange(n_subsequences):
        _dist = manhattan_distance(X_subs[i_sub], values[:, :length])
        if _dist < _min:
            _min = _dist
            _argmin = i_sub
        if _dist < threshold:
            _SO += 1

    return np.float64(_min), np.float64(_argmin), np.float64(_SO)


@njit(fastmath=True, cache=True)
def compute_shapelet_dist_vector(X_subs, values, length):
    """Extract the features from a shapelet distance vector.

    Given a shapelet and a time series, extract three features from the resulting
    distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet
    values : array, shape (n_channels, length)
        The resulting subsequence
    threshold : float
        The threshold parameter of the shapelet

    Returns
    -------
    min, argmin, shapelet occurence
        The three computed features as float dtypes
    """
    n_subsequences = X_subs.shape[0]
    dist_vector = np.zeros(n_subsequences)
    for i_sub in prange(n_subsequences):
        dist_vector[i_sub] = manhattan_distance(X_subs[i_sub], values[:, :length])
    return dist_vector
