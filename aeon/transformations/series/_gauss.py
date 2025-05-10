"""Gaussian filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["GaussSeriesTransformer"]


from deprecated.sphinx import deprecated

from aeon.transformations.series.smoothing import GaussianFilter


# TODO: Remove in v1.3.0
@deprecated(
    version="1.2.0",
    reason="GaussSeriesTransformer is deprecated and will be removed in v1.3.0. "
    "Please use GaussianFilter from "
    "transformations.series.smoothing instead.",
    category=FutureWarning,
)
class GaussSeriesTransformer(GaussianFilter):
    """Filter a times series using Gaussian filter.

    Parameters
    ----------
    sigma : float, default=1
        Standard deviation for the Gaussian kernel.

    order : int, default=0
        An order of 0 corresponds to convolution with a Gaussian kernel.
        A positive order corresponds to convolution with that derivative of a
        Gaussian.


    Notes
    -----
    More information of the SciPy gaussian_filter1d function used
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

    References
    ----------
    .. [1] Rafael C. Gonzales and Paul Wintz. 1987.
       Digital image processing.
       Addison-Wesley Longman Publishing Co., Inc., USA.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._gauss import GaussSeriesTransformer
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> gauss = GaussSeriesTransformer(sigma=5)
    >>> X_ = gauss.fit_transform(X)
    >>> X_.shape
    (2, 100)
    """

    pass
