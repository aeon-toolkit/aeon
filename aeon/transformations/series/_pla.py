"""Piecewise Linear Approximation.

A transformer which uses Piecewise Linear Approximation algorithms.
"""

__maintainer__ = []
__all__ = ["PiecewiseLinearApproximation"]

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.transformations.series.base import BaseSeriesTransformer


class PiecewiseLinearApproximation(BaseSeriesTransformer):
    """Piecewise Linear Approximation (PLA) for time series transformation.

    Takes a univariate time series as input. Approximates a time series using
    linear regression and the sum of squares error (SSE) through an algorithm.
    The algorithms available are two offline algorithms: TopDown and BottomUp
    and two online algorithms: SlidingWindow and SWAB (Sliding Window and Bottom Up).

    Parameters
    ----------
    transformer: int or str
        The transformer to be used.
        Default transformer is swab.
        Valid transformers with their string and int:
            Sliding Window: "sliding window", 1
            Top Down: "top down" , 2
            Bottom Up: "bottom up", 3
            SWAB: "swab", 4
    max_error: float
        The maximum error valuefor the function to find before segmenting the dataset.
    buffer_size: float
        The buffer size, used only for SWAB.

    Attributes
    ----------
    segment_dense : np.array
        The endpoints of each found segment of the series for transformation.

    References
    ----------
    .. [1] Keogh, E., Chu, S., Hart, D. and Pazzani, M., 2001, November.
    An online algorithm for segmenting time series. (pp. 289-296).

    Examples
    --------
    >>> from aeon.transformations.series import PiecewiseLinearApproximation
    >>> from aeon.datasets import load_electric_devices_segmentation
    >>> ts, period_size, true_cps = load_electric_devices_segmentation()
    >>> ts.values
    >>> pla = PiecewiseLinearApproximation(0.001, transformer=3)
    >>> transformed_x = pla.fit_transform(ts)
    """

    _tags = {
        "fit_is_empty": True,
    }

    def __init__(self, max_error, transformer=4, buffer_size=None):
        self.transformer = transformer
        self.max_error = max_error
        self.buffer_size = buffer_size
        self.segment_dense = None
        super().__init__(axis=0)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X: pd.DataFrame
        y : Ignored

        Returns
        -------
        self: reference to self
        """
        if not isinstance(self.max_error, (int, float)):
            raise ValueError("Invalid max_error: it has to be a number.")
        if not isinstance(self.transformer, (int, str)):
            raise ValueError("Invalid transformer: it has to be a number or a string.")
        if not (self.buffer_size is None or isinstance(self.buffer_size, (int, float))):
            raise ValueError("Invalid buffer_size: use a number only or keep empty.")
        if isinstance(self.transformer, (str)):
            if self.transformer.lower() == "sliding window":
                self.transformer = 1
            elif self.transformer.lower() == "top down":
                self.transformer = 2
            elif self.transformer.lower() == "bottom up":
                self.transformer = 3
            elif self.transformer.lower() == "swab":
                self.transformer = 4
            else:
                raise ValueError(
                    "Invalid transformer: no transformer called ", self.transformer
                )
        elif not (1 <= self.transformer <= 4):
            raise ValueError("Invalid transformer: choose between 1-4")
        self.pla_ = PiecewiseLinearApproximation(
            max_error=self.max_error,
            transformer=self.transformer,
            buffer_size=self.buffer_size,
        )
        self.pla_.fit(X=X)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform.

        Parameters
        ----------
        X : np.ndarray
            1D time series to be transformed.
        y : ignored argument for interface compatibility.

        Returns
        -------
        np.ndarray
            1D transform of X.
        """
        results = None
        if self.transformer == 1:
            results = self._sliding_window(X)
        elif self.transformer == 2:
            results = self._top_down(X)
        elif self.transformer == 3:
            results = self._bottom_up(X)
        elif self.transformer == 4:
            results = self._SWAB(X)
        else:
            raise RuntimeError("No transformer was called.")

        if len(results) > 1:
            segment_dense = np.zeros([len(results) - 1])
            segment_dense[0] = len(results[0])
            for i in range(1, len(results) - 1):
                segment_dense[i] = segment_dense[i - 1] + len(results[i])
            self.segment_dense = segment_dense

        return np.concatenate(results)

    def _sliding_window(self, X):
        """Transform a time series using the sliding window algorithm (Online).

        Parameters
        ----------
        X : np.ndarray
            1D time series to be transformed.

        Returns
        -------
        list
            List of transformed segmented time series.
        """
        seg_ts = []
        anchor = 0
        while anchor < len(X):
            i = 2
            while (
                anchor + i - 1 < len(X)
                and self._calculate_error(X[anchor : anchor + i]) < self.max_error
            ):
                i = i + 1
            seg_ts.append(self._create_segment(X[anchor : anchor + i - 1]))
            anchor = anchor + i - 1
        return seg_ts

    def _top_down(self, X):
        """Transform a time series using the top down algorithm (Offline).

        Parameters
        ----------
        X : np.ndarray
            1D time series to be transformed.

        Returns
        -------
        list
            List of transformed segmented time series.
        """
        best_so_far = float("inf")
        breakpoint = None

        for i in range(2, len(X - 2)):
            improvement_in_approximation = self.improvement_splitting_here(X, i)
            if improvement_in_approximation < best_so_far:
                breakpoint = i
                best_so_far = improvement_in_approximation

        if breakpoint is None:
            return X

        left_found_segment = X[:breakpoint]
        right_found_segment = X[breakpoint:]

        left_segment = None
        right_segment = None

        if self._calculate_error(left_found_segment) > self.max_error:
            left_segment = self._top_down(left_found_segment)
        else:
            left_segment = [self._create_segment(left_found_segment)]

        if self._calculate_error(right_found_segment) > self.max_error:
            right_segment = self._top_down(right_found_segment)
        else:
            right_segment = [self._create_segment(right_found_segment)]

        return left_segment + right_segment

    def improvement_splitting_here(self, X, breakpoint):
        """Return the SSE of two segments split at a particual point in a time series.

        Parameters
        ----------
        X : np.array
            1D time series.
        breakpoint : int
            the break point within the time series array

        Returns
        -------
        error: float
            the squared sum error of the split segmentations
        """
        left_segment = X[:breakpoint]
        right_segment = X[breakpoint:]
        return self._calculate_error(left_segment) + self._calculate_error(
            right_segment
        )

    def _bottom_up(self, X):
        """Transform a time series using the bottom up algorithm (Offline).

        Parameters
        ----------
        X : np.ndarray
            1D time series to be transformed.

        Returns
        -------
        list
            List of transformed segmented time series.
        """
        seg_ts = []
        merge_cost = []
        for i in range(0, len(X), 2):
            seg_ts.append(self._create_segment(X[i : i + 2]))
        for i in range(len(seg_ts) - 1):
            merge_cost.append(self._calculate_error(seg_ts[i] + seg_ts[i + 1]))

        merge_cost = np.array(merge_cost)

        while len(merge_cost) != 0 and min(merge_cost) < self.max_error:
            pos = np.argmin(merge_cost)
            seg_ts[pos] = self._create_segment(
                np.concatenate((seg_ts[pos], seg_ts[pos + 1]))
            )
            seg_ts.pop(pos + 1)
            if (pos + 1) < len(merge_cost):
                merge_cost = np.delete(merge_cost, pos + 1)
            else:
                merge_cost = np.delete(merge_cost, pos)

            if pos != 0:
                merge_cost[pos - 1] = self._calculate_error(
                    np.concatenate((seg_ts[pos - 1], seg_ts[pos]))
                )

            if (pos + 1) < len(seg_ts):
                merge_cost[pos] = self._calculate_error(
                    np.concatenate((seg_ts[pos], seg_ts[pos + 1]))
                )

        return seg_ts

    def _SWAB(self, X):
        """Transform a time series using the SWAB algorithm (Online).

        Parameters
        ----------
        X : np.array
            1D time series to be transformed.

        Returns
        -------
        list
            List of transformed segmented time series.
        """
        seg_ts = []
        if self.buffer_size is None:
            self.buffer_size = int(len(X) ** 0.5)

        lower_boundary_window = int(self.buffer_size / 2)
        upper_boundary_window = int(self.buffer_size * 2)

        seg = self._best_line(X, 0, lower_boundary_window, upper_boundary_window)
        current_data_point = len(seg)
        buffer = np.array(seg)

        while len(buffer) > 0:
            t = self._bottom_up(X)
            seg_ts.append(t[0])
            buffer = buffer[len(t[0]) :]
            if current_data_point >= len(X):
                seg = self._best_line(
                    X, current_data_point, lower_boundary_window, upper_boundary_window
                )
                current_data_point = current_data_point + len(seg)
                buffer = np.append(buffer, seg)
            else:
                buffer = np.array([])
                t = t[1:]
                for i in range(len(t)):
                    seg_ts.append(t[i])
        return seg_ts

    def _best_line(
        self, X, current_data_point, lower_boundary_window, upper_boundary_window
    ):
        """Use sliding window to find the next best segmentation candidate.

        Used inside of the SWAB algorithm.

        Parameters
        ----------
        X : np.array
            1D time series to be segmented.
        current_data_point : int
            the current_data_point we are observing
        lower_boundary_window: int
            the lower boundary of the window
        upper_boundary_window: int
            the uppoer boundary of the window

        Returns
        -------
        np.array
            new found segmentation candidates
        """
        max_window_length = current_data_point + upper_boundary_window
        seg_ts = np.array(
            X[current_data_point : current_data_point + lower_boundary_window]
        )
        current_data_point = current_data_point + lower_boundary_window
        error = 0
        while (
            current_data_point < max_window_length
            and current_data_point < len(X)
            and error < self.max_error
        ):
            seg_ts = np.append(seg_ts, X[current_data_point])
            error = self._calculate_error(seg_ts)
            current_data_point = current_data_point + 1
        return seg_ts

    # Create own linear regression, inefficient to use sklearns
    def _linear_regression(self, time_series):
        """Create a new time series using linear regression.

        Parameters
        ----------
        time_series : np.array
            1D time series to be transformed.

        Returns
        -------
        list
            List of transformed segmented time series
        """
        n = len(time_series)
        Y = np.array(time_series)
        X = np.arange(n).reshape(-1, 1)
        linearRegression = LinearRegression()
        linearRegression.fit(X, Y)
        regression_line = np.array(linearRegression.predict(X))
        return regression_line

    def _sum_squared_error(self, X, p_X):
        """Return the SSE of a value and its predicted value.

        formula: SSE = ∑i (Xi - p_Xi)^2

        Parameters
        ----------
        X : np.array
            1D time series.
        p_X: np.array
            1D linear time series formatted using linear regression.

        Returns
        -------
        error: float
            the SSE.
        """
        error = np.sum((X - p_X) ** 2)
        return error

    def _calculate_error(self, X):
        """Return the SEE of a time series and its linear regression.

        Parameters
        ----------
        X : np.array
            1D time series.

        Returns
        -------
        error: float
            the SSE.
        """
        lrts = self._linear_regression(X)
        sse = self._sum_squared_error(X, lrts)
        return sse

    def _create_segment(self, X):
        """Create a linear segment of a given time series.

        Parameters
        ----------
        X : np.array
            1D time series.

        Returns
        -------
        np.array
            the linear regression of the time series.
        """
        return self._linear_regression(X)
