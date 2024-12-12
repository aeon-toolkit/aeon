"""Dilated Shapelet transformers.

A modification of the classic Shapelet Transform which adds a dilation parameter to
Shapelets.
"""

__maintainer__ = ["baraline"]
__all__ = ["RandomDilatedShapeletTransform"]

import warnings
from typing import Dict
from typing import List as TypingList
from typing import Optional, Union

import numpy as np
from numba import njit, prange, set_num_threads
from numba.typed import List
from numpy.random._generator import Generator
from sklearn.preprocessing import LabelEncoder

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    choice_log,
    combinations_1d,
    get_all_subsequences,
    get_subsequence,
    get_subsequence_with_mean_std,
    normalise_subsequences,
    sliding_mean_std_one_series,
)
from aeon.utils.numba.stats import prime_up_to
from aeon.utils.validation import check_n_jobs


class RandomDilatedShapeletTransform(BaseCollectionTransformer):
    """Random Dilated Shapelet Transform (RDST) as described in [1]_, [2]_.

    Overview: The input is n series with d channels of length m. First step is to
    extract candidate shapelets from the inputs. This is done randomly, and for
    each candidate shapelet:
        - Length is randomly selected from shapelet_lengths parameter
        - Dilation is sampled as a function the shapelet length and time series length
        - Normalization is chosen randomly given the probability given as parameter
        - Start value is sampled randomly from an input time series given the length and
        dilation parameter.
        - Threshold is randomly chosen between two percentiles of the distribution
        of the distance vector between the shapelet and another time series. This time
        series is drawn from the same class if classes are given during fit. Otherwise,
        a random sample will be used. If there is only one sample per class, the same
        sample will be used.
    Then, once the set of shapelets have been initialized, we extract the shapelet
    features from each pair of shapelets and input series. Three features are extracted:
        - min d(S,X): the minimum value of the distance vector between a shapelet S and
        a time series X.
        - argmin d(S,X): the location of the minumum.
        - SO(d(S,X), threshold): The number of points in the distance vector that are
        bellow the threshold parameter of the shapelet.

    Parameters
    ----------
    max_shapelets : int, default=10000
        The maximum number of shapelets to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity has discarded the
        whole dataset.
    shapelet_lengths : array, default=None
        The set of possible lengths for shapelets. Each shapelet length is uniformly
        drawn from this set. If None, the shapelet length will be equal to
        min(max(2,n_timepoints//2),11).
    proba_normalization : float, default=0.8
        This probability (between 0 and 1) indicates the chance of each shapelet to be
        initialized such as it will use a z-normalised distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalised distance.
    threshold_percentiles : array, default=None
        The two percentiles used to select the threshold used to compute the Shapelet
        Occurrence feature. If None, the 5th and the 10th percentiles (i.e. [5,10])
        will be used.
    alpha_similarity : float, default=0.5
        The strength of the alpha similarity pruning. The higher the value, the fewer
        common indexes with previously sampled shapelets are allowed when sampling a
        new candidate with the same dilation parameter. It can cause the number of
        sampled shapelets to be lower than max_shapelets if the whole search space
        has been covered. The default is 0.5, and the maximum is 1. Values above it
        have no effect for now.
    use_prime_dilations : bool, default=False
        If True, restricts the value of the shapelet dilation parameter to be prime
        values. This can greatly speed up the algorithm for long time series and/or
        short shapelet lengths, possibly at the cost of some accuracy.
    n_jobs : int, default=1
        The number of threads used for both `fit` and `transform`.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    shapelets : list
        The stored shapelets. Each item in the list is a tuple containing:
            - shapelet values
            - startpoint values
            - length parameter
            - dilation parameter
            - threshold parameter
            - normalization parameter
            - mean parameter
            - standard deviation parameter
            - class value
    max_shapelet_length_ : int
        The maximum actual shapelet length fitted to train data.
    min_n_timepoints_ : int
        The minimum length of series in train data.

    Notes
    -----
    This implementation uses all the features for multivariate shapelets, without
    affecting a random feature subset to each shapelet as done in the original
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
    >>> X_train, y_train = load_unit_test(split="train")
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
        "capability:multithreading": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        max_shapelets: int = 10_000,
        shapelet_lengths: Optional[Union[TypingList[int], np.ndarray]] = None,
        proba_normalization: float = 0.8,
        threshold_percentiles: Optional[Union[TypingList[float], np.ndarray]] = None,
        alpha_similarity: float = 0.5,
        use_prime_dilations: bool = False,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
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

    def _fit(self, X: np.ndarray, y: Optional[Union[np.ndarray, TypingList]] = None):
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
        if isinstance(self.random_state, int):
            self._random_generator = np.random.default_rng(self.random_state)
        elif self.random_state is None:
            self._random_generator = np.random.default_rng()
        else:
            raise ValueError(
                "Expected integer or None for random_state argument but got"
                f"{type(self.random_state)}"
            )

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
            self._random_generator,
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
            (self.shapelets_[2] - 1) * self.shapelets_[3]
        )

        return self

    def _transform(
        self, X: np.ndarray, y: Optional[Union[np.ndarray, TypingList]] = None
    ):
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

        X_new = dilated_shapelet_transform(
            X,
            self.shapelets_,
        )
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

            self.shapelet_lengths_ = np.array(self.shapelet_lengths_, dtype=np.int_)
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
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> "Union[Dict, TypingList[Dict]]":
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
    max_shapelets: int,
    shapelet_lengths: np.ndarray,
    proba_normalization: float,
    use_prime_dilations: bool,
    n_channels: int,
    n_timepoints: int,
    random_generator: Generator,
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
        initialized such as it will use a z-normalised distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalised distance.
    use_prime_dilations : bool
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    n_channels : int
        Number of channels of the input time series.
    n_timepoints : int
        Size of the input time series.
    random_generator :
        The random generator used for random operations.

    Returns
    -------
    values : array, shape (max_shapelets, n_channels, max(shapelet_lengths))
        An initialized (empty) value array for each shapelet
    startpoints: array, shape (max_shapelets)
        An initialized (empty) startpoint array for each shapelet
    lengths : array, shape (max_shapelets)
        The randomly initialized length of each shapelet
    dilations : array, shape (max_shapelets)
        The randomly initialized dilation of each shapelet
    threshold : array, shape (max_shapelets)
        An initialized (empty) value array for each shapelet
    normalise : array, shape (max_shapelets)
        The randomly initialized normalization indicator of each shapelet
    means : array, shape (max_shapelets, n_channels)
        Means of the shapelets
    stds : array, shape (max_shapelets, n_channels)
        Standard deviation of the shapelets
    class: array, shape (max_shapelets)
        An initialized (empty) class array for each shapelet

    """
    # Init startpoint array
    startpoints = np.zeros(max_shapelets, dtype=np.int_)
    # Init class array
    classes = np.zeros(max_shapelets, dtype=np.int_)

    # Lengths of the shapelets
    # test dtypes correctness
    lengths = shapelet_lengths[
        random_generator.integers(
            0, high=len(shapelet_lengths), size=max_shapelets
        ).astype(np.int_)
    ]
    # Upper bound values for dilations
    dilations = np.zeros(max_shapelets, dtype=np.int_)
    upper_bounds = np.log2(np.floor_divide(n_timepoints - 1, lengths - 1))

    if use_prime_dilations:
        _primes = prime_up_to(np.int_(2 ** upper_bounds.max()))
        # 1 is not prime, but it is still a valid dilation for the "prime" scheme
        primes = np.zeros((_primes.shape[0] + 1), dtype=np.int_)
        primes[0] = 1
        primes[1:] = _primes
        for i in prange(max_shapelets):
            shp_primes = primes[primes <= np.int_(2 ** upper_bounds[i])]
            dilations[i] = shp_primes[
                choice_log(shp_primes.shape[0], 1, random_generator)[0]
            ]
    else:
        for i in prange(max_shapelets):
            dilations[i] = np.int_(2 ** random_generator.uniform(0, upper_bounds[i]))

    # Init threshold array
    threshold = np.zeros(max_shapelets, dtype=np.float64)

    # Init values array
    values = np.full(
        (max_shapelets, n_channels, max(shapelet_lengths)),
        np.inf,
        dtype=np.float64,
    )

    # Is shapelet using z-normalization ?
    normalise = random_generator.uniform(0, 1, size=max_shapelets)
    normalise = normalise < proba_normalization

    means = np.zeros((max_shapelets, n_channels), dtype=np.float64)
    stds = np.zeros((max_shapelets, n_channels), dtype=np.float64)

    return (
        values,
        startpoints,
        lengths,
        dilations,
        threshold,
        normalise,
        means,
        stds,
        classes,
    )


@njit(cache=True)
def _get_admissible_sampling_point(current_mask, random_generator):
    n_cases = len(current_mask)
    # Count the number of admissible points per sample as cumsum
    n_admissible_points = 0
    for i in range(n_cases):
        n_admissible_points += current_mask[i].shape[0]
    if n_admissible_points > 0:
        idx_choice = random_generator.integers(0, high=n_admissible_points)
        for i in range(n_cases):
            _new_val = idx_choice - current_mask[i].shape[0]
            if _new_val < 0 and current_mask[i].shape[0] > 0:
                return i, idx_choice
            idx_choice = _new_val
    else:
        return -1, -1


@njit(fastmath=True, cache=True, parallel=True)
def random_dilated_shapelet_extraction(
    X: np.ndarray,
    y: np.ndarray,
    max_shapelets: int,
    shapelet_lengths: np.ndarray,
    proba_normalization: float,
    threshold_percentiles: np.ndarray,
    alpha_similarity: float,
    use_prime_dilations: bool,
    random_generator: Generator,
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
        initialized such as it will use a z-normalised distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalised distance.
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
    random_generator : Generator
        The random generator used for random operations.

    Returns
    -------
    Shapelets : tuple
    The returned tuple contains 7 arrays describing the shapelets parameters:
        - values : array, shape (max_shapelets, n_channels, max(shapelet_lengths))
            Values of the shapelets.
        - startpoints : array, shape (max_shapelets)
            Start points parameter of the shapelets
        - lengths : array, shape (max_shapelets)
            Length parameter of the shapelets
        - dilations : array, shape (max_shapelets)
            Dilation parameter of the shapelets
        - threshold : array, shape (max_shapelets)
            Threshold parameter of the shapelets
        - normalise : array, shape (max_shapelets)
            Normalization indicator of the shapelets
        - means : array, shape (max_shapelets, n_channels)
            Means of the shapelets
        - stds : array, shape (max_shapelets, n_channels)
            Standard deviation of the shapelets
        - classes : array, shape (max_shapelets)
        An initialized (empty) startpoint array for each shapelet
    """
    n_cases = len(X)
    n_channels = X[0].shape[0]
    n_timepointss = np.zeros(n_cases, dtype=np.int_)
    for i in range(n_cases):
        n_timepointss[i] = X[i].shape[1]
    min_n_timepoints = n_timepointss.min()
    max_n_timepoints = n_timepointss.max()

    # Initialize shapelets
    (
        values,
        startpoints,
        lengths,
        dilations,
        threshold,
        normalise,
        means,
        stds,
        classes,
    ) = _init_random_shapelet_params(
        max_shapelets,
        shapelet_lengths,
        proba_normalization,
        use_prime_dilations,
        n_channels,
        min_n_timepoints,
        random_generator,
    )
    # Get unique dilations to loop over
    unique_dil = np.unique(dilations)
    n_dilations = unique_dil.shape[0]
    # For each dilation, we can do in parallel
    for i_dilation in prange(n_dilations):
        # (2, _, _): Mask is different for normalised and non-normalised shapelets
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
            norm = np.int_(normalise[i_shp])
            # Possible sampling points given self similarity mask
            current_mask = List(
                [
                    np.where(
                        alpha_mask[
                            norm,
                            _i,
                            : n_timepointss[_i] - (length - 1) * dilation,
                        ]
                    )[0]
                    for _i in range(n_cases)
                ]
            )
            idx_sample, idx_timestamp = _get_admissible_sampling_point(
                current_mask, random_generator
            )
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
                    id_test = loc_others[
                        random_generator.integers(0, high=loc_others.shape[0])
                    ]
                else:
                    id_test = idx_sample

                # Compute distance vector, first get the subsequences
                X_subs = get_all_subsequences(X[id_test], length, dilation)
                if norm:
                    # normalise them if needed
                    X_means, X_stds = sliding_mean_std_one_series(
                        X[id_test], length, dilation
                    )
                    X_subs = normalise_subsequences(X_subs, X_means, X_stds)
                x_dist = compute_shapelet_dist_vector(X_subs, _val)

                lower_bound = np.percentile(x_dist, threshold_percentiles[0])
                upper_bound = np.percentile(x_dist, threshold_percentiles[1])

                threshold[i_shp] = random_generator.uniform(lower_bound, upper_bound)
                values[i_shp, :, :length] = _val

                # Extract the starting point index of the shapelet
                startpoints[i_shp] = idx_timestamp

                # Extract the class value of the shapelet
                classes[i_shp] = y[idx_sample]

                if norm:
                    means[i_shp] = _means
                    stds[i_shp] = _stds

    mask_values = np.ones(max_shapelets, dtype=np.bool_)
    for i in prange(max_shapelets):
        if np.all(values[i] == np.inf):
            mask_values[i] = False

    return (
        values[mask_values],
        startpoints[mask_values],
        lengths[mask_values],
        dilations[mask_values],
        threshold[mask_values],
        normalise[mask_values],
        means[mask_values],
        stds[mask_values],
        classes[mask_values],
    )


@njit(fastmath=True, cache=True, parallel=True)
def dilated_shapelet_transform(
    X: np.ndarray,
    shapelets: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
):
    """Perform the shapelet transform with a set of shapelets and a set of time series.

    Parameters
    ----------
    X : array, shape (n_cases, n_channels, n_timepoints)
        Time series dataset
    shapelets : tuple
        The returned tuple contains 7 arrays describing the shapelets parameters:
        - values : array, shape (n_shapelets, n_channels, max(shapelet_lengths))
            Values of the shapelets.
        - startpoints : array, shape (max_shapelets)
            Start points parameter of the shapelets
        - lengths : array, shape (n_shapelets)
            Length parameter of the shapelets
        - dilations : array, shape (n_shapelets)
            Dilation parameter of the shapelets
        - threshold : array, shape (n_shapelets)
            Threshold parameter of the shapelets
        - normalise : array, shape (n_shapelets)
            Normalization indicator of the shapelets
        - means : array, shape (n_shapelets, n_channels)
            Means of the shapelets
        - stds : array, shape (n_shapelets, n_channels)
            Standard deviation of the shapelets
        - classes : array, shape (max_shapelets)
        An initialized (empty) startpoint array for each shapelet


    Returns
    -------
    X_new : array, shape=(n_cases, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        features from the distance vector computed on each time series.

    """
    (
        values,
        startpoints,
        lengths,
        dilations,
        threshold,
        normalise,
        means,
        stds,
        classes,
    ) = shapelets
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
            idx_no_norm = id_shps[np.where(~normalise[id_shps])[0]]
            for i_shp in idx_no_norm:
                X_new[i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)] = (
                    compute_shapelet_features(X_subs, values[i_shp], threshold[i_shp])
                )

            idx_norm = id_shps[np.where(normalise[id_shps])[0]]
            if len(idx_norm) > 0:
                X_means, X_stds = sliding_mean_std_one_series(X[i_x], length, dilation)
                X_subs = normalise_subsequences(X_subs, X_means, X_stds)
                for i_shp in idx_norm:
                    X_new[i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)] = (
                        compute_shapelet_features(
                            X_subs, values[i_shp], threshold[i_shp]
                        )
                    )
    return X_new


@njit(fastmath=True, cache=True)
def compute_shapelet_features(
    X_subs: np.ndarray,
    values: np.ndarray,
    threshold: float,
):
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

    n_subsequences, n_channels, length = X_subs.shape

    for i_sub in prange(n_subsequences):
        _dist = 0
        for k in prange(n_channels):
            for i_len in prange(length):
                _dist += abs(X_subs[i_sub, k, i_len] - values[k, i_len])
        if _dist < _min:
            _min = _dist
            _argmin = i_sub
        if _dist < threshold:
            _SO += 1

    return np.float64(_min), np.float64(_argmin), np.float64(_SO)


@njit(fastmath=True, cache=True)
def compute_shapelet_dist_vector(
    X_subs: np.ndarray,
    values: np.ndarray,
):
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
    distance: CPUDispatcher
        A Numba function used to compute the distance between two multidimensional
        time series of shape (n_channels, length).

    Returns
    -------
    dist_vector : array, shape = (n_timestamps-(length-1)*dilation)
        The distance vector between the shapelets and candidate subsequences
    """
    n_subsequences, n_channels, length = X_subs.shape
    dist_vector = np.zeros(n_subsequences)
    for i_sub in prange(n_subsequences):
        for k in prange(n_channels):
            for i_len in prange(length):
                dist_vector[i_sub] += abs(X_subs[i_sub, k, i_len] - values[k, i_len])
    return dist_vector
