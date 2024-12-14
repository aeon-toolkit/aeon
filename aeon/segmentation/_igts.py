"""
Information Gain-based Temporal Segmenter.

Information Gain Temporal Segmentation (_IGTS) is a method for segmenting
multivariate time series based off reducing the entropy in each segment [1]_.

The amount of entropy lost by the segmentations made is called the Information
Gain (IG). The aim is to find the segmentations that have the maximum information
gain for any number of segmentations.

References
----------
.. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
    "Information gain-based metric for recognizing transitions in human activities.",
    Pervasive and Mobile Computing, 38, 92-109, (2017).
    https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081

"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from aeon.segmentation.base import BaseSegmenter

__all__ = ["InformationGainSegmenter"]
__maintainer__ = []


@dataclass
class ChangePointResult:
    k: int
    score: float
    change_points: list[int]


def entropy(X: npt.ArrayLike) -> float:
    """Shannons's entropy for time series.

    As defined by equations (3) and (4) from [1]_.

    Parameters
    ----------
    X: array_like
        Time series data as a 2D numpy array with sequence index along rows
        and value series in columns.

    Returns
    -------
    entropy: float
        Computed entropy.
    """
    p = np.sum(X, axis=0) / np.sum(X)
    p = p[p > 0.0]
    return -np.sum(p * np.log(p))


def generate_segments(X: npt.ArrayLike, change_points: list[int]) -> npt.ArrayLike:
    """Generate separate segments from time series based on change points.

    Parameters
    ----------
    X: array_like
        Time series data as a 2D numpy array with sequence index along rows
        and value series in columns.

    change_points: list of int
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.

    Yields
    ------
    segment: npt.ArrayLike
        A segments from the input time series between two consecutive change points
    """
    for start, end in zip(change_points[:-1], change_points[1:]):
        yield X[start:end, :]


def generate_segments_pandas(X: npt.ArrayLike, change_points: list) -> npt.ArrayLike:
    """Generate separate segments from time series based on change points.

    Parameters
    ----------
    X: array_like
        Time series data as a 2D numpy array with sequence index along rows
        and value series in columns.

    change_points: list of int
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.

    Yields
    ------
    segment: npt.ArrayLike
        A segments from the input time series between two consecutive change points
    """
    for interval in pd.IntervalIndex.from_breaks(sorted(change_points), closed="both"):
        yield X[interval.left : interval.right, :]


@dataclass
class _IGTS:
    """
    Information Gain based Temporal Segmentation (GTS).

    GTS is a n unsupervised method for segmenting multivariate time series
    into non-overlapping segments by locating change points that for which
    the information gain is maximized.

    Information gain (IG) is defined as the amount of entropy lost by the segmentation.
    The aim is to find the segmentation that have the maximum information
    gain for a specified number of segments.

    GTS uses top-down search method to greedily find the next change point
    location that creates the maximum information gain. Once this is found, it
    repeats the process until it finds `k_max` splits of the time series.

    .. note::

       GTS does not work very well for univariate series but it can still be
       used if the original univariate series are augmented by an extra feature
       dimensions. A technique proposed in the paper [1]_ us to subtract the
       series from it's largest element and append to the series.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find. The number of segments is thus k+1.
    step: : int, default=5
        Step size, or stride for selecting candidate locations of change points.
        Fox example a `step=5` would produce candidates [0, 5, 10, ...]. Has the same
        meaning as `step` in `range` function.

    Attributes
    ----------
    intermediate_results_: list of `ChangePointResult`
        Intermediate segmentation results for each k value, where k=1, 2, ..., k_max

    Notes
    -----
    Based on the work from [1]_.
    - alt. py implementation: https://github.com/cruiseresearchgroup/IGTS-python
    - MATLAB version: https://github.com/cruiseresearchgroup/IGTS-matlab
    - paper available at:

    References
    ----------
    .. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
       "Information gain-based metric for recognizing transitions in human activities.",
       Pervasive and Mobile Computing, 38, 92-109, (2017).
       https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081
    """

    # init attributes
    k_max: int = 10
    step: int = 5

    # computed attributes
    intermediate_results_: list = field(init=False, default_factory=list)

    def identity(self, X: npt.ArrayLike) -> list[int]:
        """Return identity segmentation, i.e. terminal indexes of the data."""
        return sorted([0, X.shape[0]])

    def get_candidates(self, n_samples: int, change_points: list[int]) -> list[int]:
        """Generate candidate change points.

        Also exclude existing change points.

        Parameters
        ----------
        n_samples: int
            Length of the time series.
        change_points: list of ints
            Current set of change points, that will be used to exclude values
            from candidates.

        TODO: exclude points within a neighborhood of existing
        change points with neighborhood radius
        """
        return sorted(
            set(range(0, n_samples, self.step)).difference(set(change_points))
        )

    @staticmethod
    def information_gain_score(X: npt.ArrayLike, change_points: list[int]) -> float:
        """Calculate the information gain score.

        The formula is based on equation (2) from [1]_.

        Parameters
        ----------
        X: array_like
            Time series data as a 2D numpy array with sequence index along rows
            and value series in columns.

        change_points: list of ints
            Locations of change points as integer indexes. By convention change points
            include the identity segmentation, i.e. first and last index + 1 values.

        Returns
        -------
        information_gain: float
            Information gain score for the segmentation corresponding to the change
            points.
        """
        segment_entropies = [
            seg.shape[0] * entropy(seg) for seg in generate_segments(X, change_points)
        ]
        return entropy(X) - sum(segment_entropies) / X.shape[0]

    def find_change_points(self, X: npt.ArrayLike) -> list[int]:
        """Find change points.

        Using a top-down search method, iteratively identify at most
        `k_max` change points that increase the information gain score
        the most.

        Parameters
        ----------
        X: array_like
            Time series data as a 2D numpy array with sequence index along rows
            and value series in columns.

        Returns
        -------
        change_points: list of ints
            Locations of change points as integer indexes. By convention change points
            include the identity segmentation, i.e. first and last index + 1 values.
        """
        n_samples, n_series = X.shape
        if n_series == 1:
            raise ValueError(
                "Detected univariate series, GTS will not work properly"
                " in this case. Consider augmenting your series to multivariate."
            )
        self.intermediate_results_ = []

        # by convention initialize with the identity segmentation
        current_change_points = self.identity(X)

        for k in range(self.k_max):
            best_candidate = -1
            ig_max = -1
            # find a point which maximizes score
            for candidate in self.get_candidates(n_samples, current_change_points):
                try_change_points = {candidate}
                try_change_points.update(current_change_points)
                try_change_points = sorted(try_change_points)
                ig = self.information_gain_score(X, try_change_points)
                if ig > ig_max:
                    ig_max = ig
                    best_candidate = candidate

            current_change_points.append(best_candidate)
            current_change_points.sort()
            self.intermediate_results_.append(
                ChangePointResult(
                    k=k, score=ig_max, change_points=current_change_points.copy()
                )
            )

        return current_change_points


class InformationGainSegmenter(BaseSegmenter):
    """Information Gain based Temporal Segmentation (GTS) Estimator.

    GTS is a n unsupervised method for segmenting multivariate time series
    into non-overlapping segments by locating change points that for which
    the information gain is maximized.

    Information gain (IG) is defined as the amount of entropy lost by the segmentation.
    The aim is to find the segmentation that have the maximum information
    gain for a specified number of segments.

    GTS uses top-down search method to greedily find the next change point
    location that creates the maximum information gain. Once this is found, it
    repeats the process until it finds `k_max` splits of the time series.

    .. note::

       GTS does not work very well for univariate series but it can still be
       used if the original univariate series are augmented by an extra feature
       dimensions. A technique proposed in the paper [1]_ us to subtract the
       series from it's largest element and append to the series.

    Parameters
    ----------
    k_max: int, default=10
        Maximum number of change points to find. The number of segments is thus k+1.

    step: : int, default=5
        Step size, or stride for selecting candidate locations of change points.
        Fox example a `step=5` would produce candidates [0, 5, 10, ...]. Has the same
        meaning as `step` in `range` function.

    Attributes
    ----------
    change_points_: list of int
        Locations of change points as integer indexes. By convention change points
        include the identity segmentation, i.e. first and last index + 1 values.

    intermediate_results_: list of `ChangePointResult`
        Intermediate segmentation results for each k value, where k=1, 2, ..., k_max

    Notes
    -----
    Based on the work from [1]_.
    - alt. py implementation: https://github.com/cruiseresearchgroup/IGTS-python
    - MATLAB version: https://github.com/cruiseresearchgroup/IGTS-matlab
    - paper available at:

    References
    ----------
    .. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
       "Information gain-based metric for recognizing transitions in human activities.",
       Pervasive and Mobile Computing, 38, 92-109, (2017).
       https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_dataframe_series
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from aeon.segmentation import InformationGainSegmenter
    >>> X = make_example_dataframe_series(n_channels=2, random_state=10)
    >>> X_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    >>> igts = InformationGainSegmenter(k_max=3, step=2)
    >>> y = igts.fit_predict(X_scaled, axis=0)
    """

    _tags = {
        "capability:univariate": False,
        "capability:multivariate": True,
        "returns_dense": False,
    }

    def __init__(
        self,
        k_max: int = 10,
        step: int = 5,
    ):
        self.k_max = k_max
        self.step = step
        self._igts = _IGTS(
            k_max=k_max,
            step=step,
        )
        super().__init__(axis=0, n_segments=k_max + 1)

    def _predict(self, X, y=None) -> np.ndarray:
        """Perform segmentation.

        Parameters
        ----------
        X: np.ndarray
            2D time series shape (n_timepoints, n_channels).

        Returns
        -------
        y_pred : np.ndarray
            1D array with predicted segmentation of the same size as X.
            The numerical values represent distinct segment labels for each of the
            data points.
        """
        self.change_points_ = self._igts.find_change_points(X)
        self.intermediate_results_ = self._igts.intermediate_results_
        return self.to_clusters(self.change_points_[1:-1], X.shape[0])

    def __repr__(self) -> str:
        """Return a string representation of the estimator."""
        return self._igts.__repr__()

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
        params : dict or list of dict
        """
        return {"k_max": 2, "step": 1}
