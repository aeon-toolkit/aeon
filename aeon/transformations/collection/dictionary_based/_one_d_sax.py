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
      ``0.03 / segment_length`` for ``alphabet_size_slope``. These Gaussian
      breakpoints assume that the input time series is z-normalized (as done
      when ``znormalized=False``; see Notes). When the series length is not
      divisible by ``n_segments``, segments are partitioned via
      ``numpy.array_split`` and slope breakpoints use each segment's own length.

    The composite symbol for each segment is computed as:
    ``composite_symbol = avg_symbol * alphabet_size_slope + slope_symbol``
    producing integer symbols in the range
    ``[0, alphabet_size_avg * alphabet_size_slope - 1]``.

    A composite symbol can be decoded back into its mean and slope symbols via:
    ``avg_symbol = composite_symbol // alphabet_size_slope``
    ``slope_symbol = composite_symbol % alphabet_size_slope``

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
        Gaussian breakpoints for both segment average and slope assume
        z-normalized inputs.

    Notes
    -----
    The Gaussian breakpoints for quantizing segment slopes (variance 0.03 / l)
    and segment averages assume that the input series is z-normalized. If the
    input series is not already z-normalized, set ``znormalized=False`` (default
    is True) to z-normalize each series independently prior to segmentation.

    References
    ----------
    .. [1] Simon Malinowski, Thomas Guyet, René Quiniou, Romain Tavenard.
       1d-SAX: A Novel Symbolic Representation for Time Series.
       International Symposium on Intelligent Data Analysis, 2013.

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

        super().__init__()

    def _check_parameters(self, n_timepoints: int):
        if (
            not isinstance(self.n_segments, int)
            or isinstance(self.n_segments, bool)
            or self.n_segments < 1
        ):
            raise ValueError(
                f"n_segments must be an integer >= 1, but got {self.n_segments}"
            )
        if (
            not isinstance(self.alphabet_size_avg, int)
            or isinstance(self.alphabet_size_avg, bool)
            or self.alphabet_size_avg < 2
        ):
            raise ValueError(
                "alphabet_size_avg must be an integer >= 2, but got "
                f"{self.alphabet_size_avg}"
            )
        if (
            not isinstance(self.alphabet_size_slope, int)
            or isinstance(self.alphabet_size_slope, bool)
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
        if self.n_segments > n_timepoints:
            raise ValueError(
                f"n_segments ({self.n_segments}) cannot be greater than "
                f"the series length ({n_timepoints})."
            )

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

        self._check_parameters(n_timepoints)

        breakpoints_avg = scipy.stats.norm.ppf(
            np.arange(1, self.alphabet_size_avg, dtype=np.float64)
            / self.alphabet_size_avg
        )

        if not self.znormalized:
            X = SAX._z_normalize(X)

        if n_timepoints % self.n_segments == 0:
            seg_len = n_timepoints // self.n_segments
            X_reshaped = X.reshape(n_cases, n_channels, self.n_segments, seg_len)
            avg_symbols, slope_symbols = _quantize_segment(
                X_reshaped, breakpoints_avg, self.alphabet_size_slope
            )
        else:
            split_indices = np.array_split(np.arange(n_timepoints), self.n_segments)
            avg_symbols = np.empty(
                (n_cases, n_channels, self.n_segments), dtype=np.int64
            )
            slope_symbols = np.empty(
                (n_cases, n_channels, self.n_segments), dtype=np.int64
            )

            for s, seg_idx in enumerate(split_indices):
                avg_s, slope_s = _quantize_segment(
                    X[:, :, seg_idx], breakpoints_avg, self.alphabet_size_slope
                )
                avg_symbols[:, :, s] = avg_s
                slope_symbols[:, :, s] = slope_s

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


def _quantize_segment(
    data: np.ndarray,
    breakpoints_avg: np.ndarray,
    alphabet_size_slope: int,
):
    """Compute mean and slope quantized symbols for a segment or segments.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (..., seg_len) where the last dimension is time.
    breakpoints_avg : np.ndarray
        Gaussian breakpoints for quantizing segment averages.
    alphabet_size_slope : int
        Alphabet size for quantizing segment slopes.

    Returns
    -------
    avg_symbols : np.ndarray
        Quantized mean symbols of shape (...).
    slope_symbols : np.ndarray
        Quantized slope symbols of shape (...).
    """
    seg_len = data.shape[-1]
    means = np.mean(data, axis=-1)

    if seg_len > 1:
        # Centered coordinates ensure sum(x) == 0, separating mean and slope
        x = np.arange(seg_len, dtype=np.float64) - (seg_len - 1.0) / 2.0
        sum_x2 = (seg_len * (seg_len**2 - 1.0)) / 12.0
        slopes = np.sum(data * x, axis=-1) / sum_x2
        scale_slope = np.sqrt(0.03 / seg_len)
    else:
        # When segment_length == 1, slope is undefined; 0.0 reflects flat trend
        slopes = np.zeros_like(means)
        scale_slope = np.sqrt(0.03)

    breakpoints_slope = scipy.stats.norm.ppf(
        np.arange(1, alphabet_size_slope, dtype=np.float64) / alphabet_size_slope,
        scale=scale_slope,
    )

    avg_symbols = np.digitize(means, breakpoints_avg, right=False)
    slope_symbols = np.digitize(slopes, breakpoints_slope, right=False)
    return avg_symbols, slope_symbols
