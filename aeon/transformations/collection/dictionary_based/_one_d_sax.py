"""One-Dimensional Symbolic Aggregate Approximation (1d-SAX) transformer."""

__maintainer__ = []
__all__ = ["OneDSAX"]

import numpy as np
import scipy.stats

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.collection.dictionary_based._sax import SAX


class OneDSAX(BaseCollectionTransformer):
    """One-Dimensional Symbolic Aggregate Approximation (1d-SAX) transformer.

    1d-SAX [1]_ extends standard SAX by modeling each time series segment with
    both its mean (PAA average) and linear trend (slope). Each segment's mean
    and slope are independently quantized into discrete symbols using Gaussian
    breakpoints:
    - Mean symbols are derived from standard normal quantiles for
      ``alphabet_size_avg``.
    - Slope symbols are derived from normal quantiles with variance
      ``0.03 / segment_length`` for ``alphabet_size_slope``. When the series
      length is not divisible by ``n_segments``, segments are partitioned via
      ``numpy.array_split`` and slope breakpoints use each segment's own length.

    The composite symbol for each segment is computed as:
    ``symbol = avg_symbol * alphabet_size_slope + slope_symbol``
    producing integer symbols in the range
    ``[0, alphabet_size_avg * alphabet_size_slope - 1]``.

    Parameters
    ----------
    n_segments : int, default = 8
        Number of segments into which each time series is divided.
    alphabet_size_avg : int, default = 4
        Size of the alphabet used to quantize the segment average (mean).
    alphabet_size_slope : int, default = 4
        Size of the alphabet used to quantize the segment slope.
    znormalized : bool, default = True
        Whether the input series is already z-normalized. If False, each series
        is z-normalized independently along the time axis prior to segmentation.

    References
    ----------
    .. [1] Malinowski, A. L., Bhasin, H. S., Kawasaki, T., & Furukawa, T. (2013).
       1d-SAX: a novel symbolic representation for time series.
       Knowledge and Information Systems, 37(1), 175-198.

    Examples
    --------
    >>> from aeon.transformations.collection.dictionary_based import OneDSAX
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> od_sax = OneDSAX(n_segments=4, alphabet_size_avg=3, alphabet_size_slope=3)
    >>> X_transformed = od_sax.fit_transform(X_train)
    >>> X_transformed.shape
    (20, 1, 4)
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        n_segments: int = 8,
        alphabet_size_avg: int = 4,
        alphabet_size_slope: int = 4,
        znormalized: bool = True,
    ):
        self.n_segments = n_segments
        self.alphabet_size_avg = alphabet_size_avg
        self.alphabet_size_slope = alphabet_size_slope
        self.znormalized = znormalized

        if not isinstance(self.n_segments, int) or self.n_segments < 1:
            raise ValueError(
                f"n_segments must be an integer >= 1, but got {self.n_segments}"
            )
        if not isinstance(self.alphabet_size_avg, int) or self.alphabet_size_avg < 2:
            raise ValueError(
                "alphabet_size_avg must be an integer >= 2, but got "
                f"{self.alphabet_size_avg}"
            )
        if (
            not isinstance(self.alphabet_size_slope, int)
            or self.alphabet_size_slope < 2
        ):
            raise ValueError(
                "alphabet_size_slope must be an integer >= 2, but got "
                f"{self.alphabet_size_slope}"
            )
        if not isinstance(self.znormalized, bool):
            raise ValueError(
                f"znormalized must be a boolean, but got {type(self.znormalized)}"
            )

        self.breakpoints_avg = scipy.stats.norm.ppf(
            np.arange(1, self.alphabet_size_avg, dtype=np.float64)
            / self.alphabet_size_avg
        )

        super().__init__()

    def _transform(self, X, y=None):
        """Transform input time series collection into 1d-SAX symbols.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The input time series collection.
        y : None
            Ignored.

        Returns
        -------
        symbols : np.ndarray of shape (n_cases, n_channels, n_segments), dtype int64
            The composite 1d-SAX symbols.
        """
        n_cases, n_channels, n_timepoints = X.shape

        if self.n_segments > n_timepoints:
            raise ValueError(
                f"n_segments ({self.n_segments}) cannot be greater than "
                f"the series length ({n_timepoints})."
            )

        if not self.znormalized:
            X = SAX._z_normalize(X)

        if n_timepoints % self.n_segments == 0:
            seg_len = n_timepoints // self.n_segments
            X_reshaped = X.reshape(n_cases, n_channels, self.n_segments, seg_len)

            means = np.mean(X_reshaped, axis=-1)

            if seg_len > 1:
                # Centered coordinates ensure sum(x) == 0, separating mean and slope
                x = np.arange(seg_len, dtype=np.float64) - (seg_len - 1.0) / 2.0
                sum_x2 = (seg_len * (seg_len**2 - 1.0)) / 12.0
                slopes = np.sum(X_reshaped * x, axis=-1) / sum_x2
                scale_slope = np.sqrt(0.03 / seg_len)
            else:
                # When segment_length == 1, slope is undefined; 0.0 reflects flat trend
                slopes = np.zeros_like(means)
                scale_slope = np.sqrt(0.03)

            breakpoints_slope = scipy.stats.norm.ppf(
                np.arange(1, self.alphabet_size_slope, dtype=np.float64)
                / self.alphabet_size_slope,
                scale=scale_slope,
            )

            avg_symbols = np.digitize(means, self.breakpoints_avg, right=False)
            slope_symbols = np.digitize(slopes, breakpoints_slope, right=False)
        else:
            split_indices = np.array_split(np.arange(n_timepoints), self.n_segments)
            avg_symbols = np.empty(
                (n_cases, n_channels, self.n_segments), dtype=np.int64
            )
            slope_symbols = np.empty(
                (n_cases, n_channels, self.n_segments), dtype=np.int64
            )

            for s, seg_idx in enumerate(split_indices):
                seg_len = len(seg_idx)
                X_s = X[:, :, seg_idx]
                means_s = np.mean(X_s, axis=-1)

                if seg_len > 1:
                    x = np.arange(seg_len, dtype=np.float64) - (seg_len - 1.0) / 2.0
                    sum_x2 = (seg_len * (seg_len**2 - 1.0)) / 12.0
                    slopes_s = np.sum(X_s * x, axis=-1) / sum_x2
                    scale_slope = np.sqrt(0.03 / seg_len)
                else:
                    slopes_s = np.zeros_like(means_s)
                    scale_slope = np.sqrt(0.03)

                breakpoints_slope = scipy.stats.norm.ppf(
                    np.arange(1, self.alphabet_size_slope, dtype=np.float64)
                    / self.alphabet_size_slope,
                    scale=scale_slope,
                )

                avg_symbols[:, :, s] = np.digitize(
                    means_s, self.breakpoints_avg, right=False
                )
                slope_symbols[:, :, s] = np.digitize(
                    slopes_s, breakpoints_slope, right=False
                )

        symbols = avg_symbols * self.alphabet_size_slope + slope_symbols
        return symbols.astype(np.int64, copy=False)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return [
            {"n_segments": 4, "alphabet_size_avg": 3, "alphabet_size_slope": 3},
            {"n_segments": 2, "alphabet_size_avg": 2, "alphabet_size_slope": 4},
        ]
