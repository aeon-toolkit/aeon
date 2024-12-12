"""ClaSP (Classification Score Profile) Segmentation."""

import warnings

__maintainer__ = []
__all__ = ["ClaSPSegmenter", "find_dominant_window_sizes"]

from queue import PriorityQueue

import numpy as np
import pandas as pd

from aeon.segmentation.base import BaseSegmenter
from aeon.transformations.series import ClaSPTransformer


def find_dominant_window_sizes(X, offset=0.05):
    """Determine the Window-Size using dominant FFT-frequencies.

    Parameters
    ----------
    X : array-like, shape=[n]
        a single univariate time series of length n
    offset : float
        Exclusion Radius

    Returns
    -------
    trivial_match: bool
        If the candidate change point is a trivial match
    """
    fourier = np.absolute(np.fft.fft(X))
    freqs = np.fft.fftfreq(X.shape[0], 1)

    coefs = []
    window_sizes = []

    for coef, freq in zip(fourier, freqs):
        if coef and freq > 0:
            coefs.append(coef)
            window_sizes.append(1 / freq)

    coefs = np.array(coefs)
    window_sizes = np.asarray(window_sizes, dtype=np.int64)

    idx = np.argsort(coefs)[::-1]
    return next(
        (
            int(window_size / 2)
            for window_size in window_sizes[idx]
            if window_size in range(20, int(X.shape[0] * offset))
        ),
        window_sizes[idx[0]],
    )


def _is_trivial_match(candidate, change_points, n_timepoints, exclusion_radius=0.05):
    """Check if a candidate change point is in close proximity to other change points.

    Parameters
    ----------
    candidate : int
        A single candidate change point. Will be chosen if non-trivial match based
        on exclusion_radius.
    change_points : list, dtype=int
        List of change points chosen so far
    n_timepoints : int
        Total length
    exclusion_radius : float
        Exclusion Radius for change points to be non-trivial matches

    Returns
    -------
    trivial_match: bool
        If the 'candidate' change point is a trivial match to the ones in change_points
    """
    change_points = [0] + change_points + [n_timepoints]
    exclusion_zone = np.int64(n_timepoints * exclusion_radius)

    for change_point in change_points:
        left_begin = max(0, change_point - exclusion_zone)
        right_end = min(n_timepoints, change_point + exclusion_zone)
        if candidate in range(left_begin, right_end):
            return True

    return False


def _segmentation(X, clasp, n_change_points=None, exclusion_radius=0.05):
    """Segments the time series by extracting change points.

    Parameters
    ----------
    X : array-like, shape=[n]
        the univariate time series of length n to be segmented
    clasp :
        the transformer
    n_change_points : int
        the number of change points to find
    exclusion_radius : float
        the exclusion zone

    Returns
    -------
    Tuple (array-like, array-like, array-like):
        (predicted_change_points, clasp_profiles, scores)
    """
    period_size = clasp.window_length
    queue = PriorityQueue()

    # compute global clasp
    profile = clasp.transform(X)
    queue.put(
        (
            -np.max(profile),
            [np.arange(X.shape[0]).tolist(), np.argmax(profile), profile],
        )
    )

    profiles = []
    change_points = []
    scores = []

    for idx in range(n_change_points):
        # should not happen ... safety first
        if queue.empty() is True:
            break

        # get profile with highest change point score
        priority, (profile_range, change_point, full_profile) = queue.get()

        change_points.append(change_point)
        scores.append(-priority)
        profiles.append(full_profile)

        if idx == n_change_points - 1:
            break

        # create left and right local range
        left_range = np.arange(profile_range[0], change_point).tolist()
        right_range = np.arange(change_point, profile_range[-1]).tolist()

        for ranges in [left_range, right_range]:
            # create and enqueue left local profile
            exclusion_zone = np.int64(len(ranges) * exclusion_radius)
            if len(ranges) - period_size > 2 * exclusion_zone:
                profile = clasp.transform(X[ranges])
                change_point = np.argmax(profile)
                score = profile[change_point]

                full_profile = np.zeros(len(X))
                full_profile.fill(0.5)
                np.copyto(
                    full_profile[ranges[0] : ranges[0] + len(profile)],
                    profile,
                )

                global_change_point = ranges[0] + change_point

                if not _is_trivial_match(
                    global_change_point,
                    change_points,
                    X.shape[0],
                    exclusion_radius=exclusion_radius,
                ):
                    queue.put((-score, [ranges, global_change_point, full_profile]))

    return np.array(change_points), np.array(profiles, dtype=object), np.array(scores)


class ClaSPSegmenter(BaseSegmenter):
    """ClaSP (Classification Score Profile) Segmentation.

    Using ClaSP [1]_, [2]_ for the CPD problem is straightforward: We first compute the
    profile and then choose its global maximum as the change point. The following CPDs
    are obtained using a bespoke recursive split segmentation algorithm.

    Parameters
    ----------
    period_length : int, default = 10
        Size of window for sliding, based on the period length of the data.
    n_cps : int, default = 1
        The number of change points to search.
    exclusion_radius : int
        Exclusion Radius for change points to be non-trivial matches.
    n_jobs : int, default=1
        Number of jobs to be used.

    References
    ----------
    .. [1] Schafer, Patrick and Ermshaus, Arik and Leser, Ulf. "ClaSP - Time Series
    Segmentation", CIKM, 2021.
    .. [2] Ermshaus, Arik, Sch"afer, Patrick and Leser, Ulf. ClaSP: parameter-free
    time series segmentation. Data Mining and Knowledge Discovery, 37, 2023.

    Examples
    --------
    >>> from aeon.segmentation import ClaSPSegmenter
    >>> from aeon.segmentation import find_dominant_window_sizes
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, true_period_size, cps = load_gun_point_segmentation()
    >>> dominant_period_size = find_dominant_window_sizes(X)
    >>> clasp = ClaSPSegmenter(dominant_period_size, n_cps=1)
    >>> found_cps = clasp.fit_predict(X)
    >>> profiles = clasp.profiles
    >>> scores = clasp.scores
    """

    _tags = {"capability:multithreading": True, "fit_is_empty": True}

    def __init__(self, period_length=10, n_cps=1, exclusion_radius=0.05, n_jobs=1):
        self.period_length = int(period_length)
        self.n_cps = n_cps
        self.exclusion_radius = exclusion_radius
        self.n_jobs = n_jobs
        super().__init__(axis=1, n_segments=n_cps + 1)

    def _predict(self, X: np.ndarray):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : np.ndarray
            1D time series to be segmented.

        Returns
        -------
        list
            List of change points found in X.
        """
        X = X.squeeze()
        if len(X) - self.period_length < 2 * self.exclusion_radius * len(X):
            warnings.warn(
                "Period-Length is larger than size of the time series", stacklevel=1
            )

            self.found_cps, self.profiles, self.scores = [], [], []
        else:
            self.found_cps, self.profiles, self.scores = self._run_clasp(X)
            return self.found_cps

    def predict_scores(self, X):
        """Return scores in ClaSP's profile for each annotation.

        Parameters
        ----------
        np.ndarray
            1D time series to be segmented.

        Returns
        -------
        np.ndarray
            Scores for sequence X
        """
        self.found_cps, self.profiles, self.scores = self._run_clasp(X)
        return self.scores

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {"profiles": self.profiles, "scores": self.scores}

    def _run_clasp(self, X):
        clasp_transformer = ClaSPTransformer(
            window_length=self.period_length,
            exclusion_radius=self.exclusion_radius,
            n_jobs=self.n_jobs,
        ).fit(X)

        self.found_cps, self.profiles, self.scores = _segmentation(
            X,
            clasp_transformer,
            n_change_points=self.n_cps,
            exclusion_radius=self.exclusion_radius,
        )

        return self.found_cps, self.profiles, self.scores

    def _get_interval_series(self, X, found_cps):
        """Get the segmentation results based on the found change points.

        Parameters
        ----------
        X :         array-like, shape = [n]
           Univariate time-series data to be segmented.
        found_cps : array-like, shape = [n_cps] The found change points found

        Returns
        -------
        IntervalIndex:
            Segmentation based on found change points

        """
        cps = np.array(found_cps)
        start = np.insert(cps, 0, 0)
        end = np.append(cps, len(X))
        return pd.IntervalIndex.from_arrays(start, end)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"period_length": 5, "n_cps": 1}
