"""Discrete Fourier Approximation filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["DFTSeriesTransformer"]


from deprecated.sphinx import deprecated

from aeon.transformations.series.smoothing import DiscreteFourierApproximation


# TODO: Remove in v1.3.0
@deprecated(
    version="1.2.0",
    reason="DFTSeriesTransformer is deprecated and will be removed in v1.3.0. "
    "Please use DiscreteFourierApproximation from "
    "transformations.series.smoothing instead.",
    category=FutureWarning,
)
class DFTSeriesTransformer(DiscreteFourierApproximation):
    """Filter a times series using Discrete Fourier Approximation (DFT).

    Parameters
    ----------
    r : float
        Proportion of Fourier terms to retain [0, 1]

    sort : bool
        Sort the Fourier terms by amplitude to keep most important terms

    Notes
    -----
    More information of the NumPy FFT functions used
    https://numpy.org/doc/stable/reference/routines.fft.html

    References
    ----------
    .. [1] Cooley, J.W., & Tukey, J.W. (1965).
       An algorithm for the machine calculation of complex Fourier series.
       Mathematics of Computation, 19, 297-301.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._dft import DFTSeriesTransformer
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> dft = DFTSeriesTransformer()
    >>> X_ = dft.fit_transform(X)
    >>> X_.shape
    (2, 100)
    """

    ...
