"""Savitzky-Golay filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["SGSeriesTransformer"]


from deprecated.sphinx import deprecated

from aeon.transformations.series.smoothing import SavitzkyGolayFilter


# TODO: Remove in v1.3.0
@deprecated(
    version="1.2.0",
    reason="SGSeriesTransformer is deprecated and will be removed in v1.3.0. "
    "Please use SavitzkyGolayFilter from "
    "transformations.series.smoothing instead.",
    category=FutureWarning,
)
class SGSeriesTransformer(SavitzkyGolayFilter):
    """Filter a times series using Savitzky-Golay (SG).

    Parameters
    ----------
    window_length : int, default=5
        The length of the filter window (i.e., the number of coefficients).
        window_length must be less than or equal to the size of the input.

    polyorder : int, default=2
        The order of the polynomial used to fit the samples.
        polyorder must be less than window_length.


    Notes
    -----
    More information of the SciPy savgol_filter function used
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    References
    ----------
    .. [1] Savitzky, A., & Golay, M. J. (1964).
       Smoothing and differentiation of data by simplified least squares procedures.
       Analytical chemistry, 36(8), 1627-1639.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._sg import SGSeriesTransformer
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> sg = SGSeriesTransformer()
    >>> X_ = sg.fit_transform(X)
    >>> X_.shape
    (2, 100)
    """

    ...
