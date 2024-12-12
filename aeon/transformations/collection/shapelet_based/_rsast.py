"""RSAST Transformer."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.numba.general import z_normalise_series
from aeon.utils.validation import check_n_jobs


@njit(fastmath=False)
def _apply_kernel(ts: np.ndarray, arr: np.ndarray) -> float:
    d_best = np.inf  # sdist
    m = ts.shape[0]
    kernel = arr[~np.isnan(arr)]  # ignore nan

    kernel_len = kernel.shape[0]
    for i in range(m - kernel_len + 1):
        d = np.sum((z_normalise_series(ts[i : i + kernel_len]) - kernel) ** 2)
        if d < d_best:
            d_best = d
    return d_best


@njit(parallel=True, fastmath=True)
def _apply_kernels(X: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        for t in range(X.shape[0]):
            ts = X[t]
            out[t][i] = _apply_kernel(ts, k)
    return out


class RSAST(BaseCollectionTransformer):
    """Random Scalable and Accurate Subsequence Transform (RSAST).

    RSAST [1] is based on SAST, it uses a stratified sampling strategy
    for subsequences selection but additionally takes into account certain
    statistical criteria such as ANOVA, ACF, and PACF to further reduce
    the search space of shapelets.

    RSAST starts with the pre-computation of a list of weights, using ANOVA,
    which helps in the selection of initial points for subsequences. Then
    randomly select k time series per class, which are used with an ACF and PACF,
    obtaining a set of highly correlated lagged values. These values are used as
    potential lengths for the shapelets. Lastly, with a pre-defined number of
    admissible starting points to sample, the shapelets are extracted and used to
    transform the original dataset, replacing each time series by the vector of its
    distance to each subsequence.

    Parameters
    ----------
    n_random_points: int default = 10
        the number of initial random points to extract
    len_method:  string default="both" the type of statistical tool used to get
    the length of shapelets. "both"=ACF&PACF, "ACF"=ACF, "PACF"=PACF,
    "None"=Extract randomly any length from the TS

    nb_inst_per_class : int default = 10
        the number of reference time series to select per class
    seed : int, default = None
        the seed of the random generator
    n_jobs : int, default -1
        Number of threads to use for the transform.

    References
    ----------
    .. [1] Varela, N. R., Mbouopda, M. F., & Nguifo, E. M. (2023).
    RSAST: Sampling Shapelets for Time Series Classification.
    https://hal.science/hal-04311309/


    Examples
    --------
    >>> from aeon.transformations.collection.shapelet_based import RSAST
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> rsast = RSAST() # doctest: +SKIP
    >>> rsast.fit(X_train, y_train) # doctest: +SKIP
    RSAST()
    >>> X_train = rsast.transform(X_train) # doctest: +SKIP
    >>> X_test = rsast.transform(X_test) # doctest: +SKIP

    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": False,
        "capability:multithreading": True,
        "algorithm_type": "shapelet",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_random_points: int = 10,
        len_method: str = "both",
        nb_inst_per_class: int = 10,
        seed: Optional[int] = None,
        n_jobs: int = 1,  # Parllel Processing
    ):
        self.n_random_points = n_random_points
        self.len_method = len_method
        self.nb_inst_per_class = nb_inst_per_class
        self.n_jobs = n_jobs
        self.seed = seed
        self._kernels = None  # z-normalized subsequences
        self._cand_length_list = {}
        self._kernel_orig = []
        self._start_points = []
        self._classes = []
        self._source_series = []  # To store the index of the original time series
        self._kernels_generators = {}  # Reference time series
        super().__init__()

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, list]) -> "RSAST":
        from scipy.stats import ConstantInputWarning, DegenerateDataWarning, f_oneway
        from statsmodels.tsa.stattools import acf, pacf

        """Select reference time series and generate subsequences from them.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : RSAST
            This transformer

        """

        # 0- initialize variables and convert values in "y" to string
        X_ = np.reshape(X, (X.shape[0], X.shape[-1]))

        self._random_state = (
            np.random.RandomState(self.seed)
            if not isinstance(self.seed, np.random.RandomState)
            else self.seed
        )

        classes = np.unique(y)
        self._num_classes = classes.shape[0]

        y = np.asarray([str(x_s) for x_s in y])

        n = []
        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        m_kernel = 0

        # Initialize lists to store start positions, classes, and source series
        self._start_points = []
        self._classes = []
        self._source_series = []

        # 1--calculate ANOVA per each time t throughout the length of the TS
        for i in range(X_.shape[1]):
            statistic_per_class = {}
            for c in classes:
                assert (
                    len(X_[np.where(y == c)[0]][:, i]) > 0
                ), "Time t without values in TS"
                statistic_per_class[c] = X_[np.where(y == c)[0]][:, i]

            statistic_per_class = pd.Series(statistic_per_class)
            # Calculate t-statistic and p-value
            try:
                t_statistic, p_value = f_oneway(*statistic_per_class)
            except (DegenerateDataWarning, ConstantInputWarning):
                p_value = np.nan

            # Interpretation of the results
            # if p_value < 0.05: " The means of the populations are
            # significantly different."
            if np.isnan(p_value):
                n.append(0)
            else:
                n.append(1 - p_value)

        # 2--calculate PACF and ACF for each TS chosen in each class

        for i, c in enumerate(classes):
            X_c = X_[y == c]

            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)

            # Store the original indices of the sampled time series
            original_indices = np.where(y == c)[0]

            chosen_indices = self._random_state.permutation(X_c.shape[0])[:cnt]

            self._kernels_generators[c] = []

            for rep, idx in enumerate(chosen_indices):
                original_idx = original_indices[idx]  # Get the original index
                # defining indices for length list
                idx_len_list = c + "," + str(idx) + "," + str(rep)

                self._cand_length_list[idx_len_list] = []

                non_zero_acf = []
                if self.len_method == "both" or self.len_method == "ACF":
                    # 2.1 -- Compute statsmodels autocorrelation per series
                    acf_val, acf_confint = acf(
                        X_c[idx], nlags=len(X_c[idx]) - 1, alpha=0.05
                    )

                    for j in range(len(acf_confint)):
                        if 3 <= j and (
                            0 < acf_confint[j][0] <= acf_confint[j][1]
                            or acf_confint[j][0] <= acf_confint[j][1] < 0
                        ):
                            non_zero_acf.append(j)
                            self._cand_length_list[idx_len_list].append(j)

                non_zero_pacf = []
                if self.len_method == "both" or self.len_method == "PACF":
                    # 2.2 Compute Partial Autocorrelation per series
                    pacf_val, pacf_confint = pacf(
                        X_c[idx],
                        method="ols",
                        nlags=(len(X_c[idx]) // 2) - 1,
                        alpha=0.05,
                    )

                    for j in range(len(pacf_confint)):
                        if 3 <= j and (
                            0 < pacf_confint[j][0] <= pacf_confint[j][1]
                            or pacf_confint[j][0] <= pacf_confint[j][1] < 0
                        ):
                            non_zero_pacf.append(j)
                            self._cand_length_list[idx_len_list].append(j)

                if self.len_method == "all":
                    self._cand_length_list[idx_len_list].extend(
                        np.arange(3, 1 + len(X_c[idx]))
                    )

                # 2.3-- Save the maximum autocorrelated lag value as shapelet length
                if len(self._cand_length_list[idx_len_list]) == 0:
                    # chose a random length using the length of the time series
                    # (added 1 since the range start in 0)
                    rand_value = self._random_state.choice(len(X_c[idx]), 1)[0] + 1
                    self._cand_length_list[idx_len_list].extend([max(3, rand_value)])

                self._cand_length_list[idx_len_list] = list(
                    set(self._cand_length_list[idx_len_list])
                )

                for max_shp_length in self._cand_length_list[idx_len_list]:
                    # 2.4-- Choose randomly n_random_points point for a TS
                    # 2.5-- calculate the weights of probabilities for a random point
                    # in a TS
                    if sum(n) == 0:
                        # Determine equal weights of a random point in TS ix
                        weights = [1 / len(n) for i in range(len(n))]
                        weights = weights[
                            : len(X_c[idx]) - max_shp_length + 1
                        ] / np.sum(weights[: len(X_c[idx]) - max_shp_length + 1])
                    else:
                        # Determine the weights of a random point in TS
                        # (excluding points after n-l+1)
                        weights = n / np.sum(n)
                        weights = weights[
                            : len(X_c[idx]) - max_shp_length + 1
                        ] / np.sum(weights[: len(X_c[idx]) - max_shp_length + 1])

                    if self.n_random_points > len(X_c[idx]) - max_shp_length + 1:
                        # set an upper limit for the possible number of random
                        # points when selecting without replacement
                        limit_rpoint = len(X_c[idx]) - max_shp_length + 1
                        rand_point_ts = self._random_state.choice(
                            len(X_c[idx]) - max_shp_length + 1,
                            limit_rpoint,
                            p=weights,
                            replace=False,
                        )
                    else:
                        rand_point_ts = self._random_state.choice(
                            len(X_c[idx]) - max_shp_length + 1,
                            self.n_random_points,
                            p=weights,
                            replace=False,
                        )

                    for i in rand_point_ts:
                        # 2.6-- Extract the subsequence with that point
                        kernel = X_c[idx][i : i + max_shp_length].reshape(1, -1).copy()

                        if m_kernel < max_shp_length:
                            m_kernel = max_shp_length

                        self._kernel_orig.append(np.squeeze(kernel))
                        self._kernels_generators[c].extend(X_c[idx].reshape(1, -1))

                        # Store the start position,
                        # class, and the original index in the training set
                        self._start_points.append(i)
                        self._classes.append(c)
                        self._source_series.append(original_idx)

        # 3--save the calculated subsequences
        n_kernels = len(self._kernel_orig)

        self._kernels = np.full(
            (n_kernels, m_kernel), dtype=np.float32, fill_value=np.nan
        )

        for k, kernel in enumerate(self._kernel_orig):
            self._kernels[k, : len(kernel)] = z_normalise_series(kernel)

        return self

    def _transform(
        self, X: np.ndarray, y: Optional[Union[np.ndarray, list]] = None
    ) -> np.ndarray:
        """Transform the input X using the generated subsequences.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            Ignored argument, interface compatibility

        Returns
        -------
        X_transformed: np.ndarray shape (n_cases, n_kernels),
            The transformed data
        """
        X_ = np.reshape(X, (X.shape[0], X.shape[-1]))

        prev_threads = get_num_threads()

        n_jobs = check_n_jobs(self.n_jobs)

        set_num_threads(n_jobs)

        X_transformed = _apply_kernels(X_, self._kernels)  # subsequence transform of X
        set_num_threads(prev_threads)

        return X_transformed
