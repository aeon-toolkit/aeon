"""Recursive Median Sieve filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["SIVSeriesTransformer"]


from deprecated.sphinx import deprecated

from aeon.transformations.series.smoothing import RecursiveMedianSieve


# TODO: Remove in v1.3.0
@deprecated(
    version="1.2.0",
    reason="SIVSeriesTransformer is deprecated and will be removed in v1.3.0. "
    "Please use RecursiveMedianSieve from "
    "transformations.series.smoothing instead.",
    category=FutureWarning,
)
class SIVSeriesTransformer(RecursiveMedianSieve):
    """Filter a times series using Recursive Median Sieve (SIV).

    Parameters
    ----------
    window_length : list of int or int, default=[3, 5, 7]
        The filter windows lengths (recommended increasing value).

    Notes
    -----
    Use scipy.ndimage.median_filter instead of scipy.signal.medfilt :
    The more general function scipy.ndimage.median_filter has a more efficient
    implementation of a median filter and therefore runs much faster.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html

    References
    ----------
    .. [1] Bangham J. A. (1988).
       Data-sieving hydrophobicity plots.
       Analytical biochemistry, 174(1), 142–145.
       https://doi.org/10.1016/0003-2697(88)90528-3
    .. [2] Yli-Harja, O., Koivisto, P., Bangham, J. A., Cawley, G.,
       Harvey, R., & Shmulevich, I. (2001).
       Simplified implementation of the recursive median sieve.
       Signal Process., 81(7), 1565–1570.
       https://doi.org/10.1016/S0165-1684(01)00054-8

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._siv import SIVSeriesTransformer
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> siv = SIVSeriesTransformer()
    >>> X_ = siv.fit_transform(X)
    >>> X_.shape
    (2, 100)
    """

    pass
