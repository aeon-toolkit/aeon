"""Moving average transformation."""

__maintainer__ = ["Datadote"]
__all__ = ["MovingAverageSeriesTransformer"]


from deprecated.sphinx import deprecated

from aeon.transformations.series.smoothing import MovingAverage


# TODO: Remove in v1.3.0
@deprecated(
    version="1.2.0",
    reason="MovingAverageSeriesTransformer is deprecated and will be removed in "
    "v1.3.0. Please use MovingAverage from "
    "transformations.series.smoothing instead.",
    category=FutureWarning,
)
class MovingAverageSeriesTransformer(MovingAverage):
    """Calculate the moving average of an array of numbers.

    Slides a window across the input array, and returns the averages for each window.
    This implementation precomputes a cumulative sum, and then performs subtraction.

    Parameters
    ----------
    window_size: int, default=5
        Number of values to average for each window

    References
    ----------
    Large, J., Southam, P., Bagnall, A. (2019).
        Can Automated Smoothing Significantly Improve Benchmark Time Series
        Classification Algorithms?. In: Pérez García, H., Sánchez González,
        L., CastejónLimas, M., Quintián Pardo, H., Corchado Rodríguez, E. (eds) Hybrid
        Artificial Intelligent Systems. HAIS 2019. Lecture Notes in Computer Science(),
        vol 11734. Springer, Cham. https://doi.org/10.1007/978-3-030-29859-3_5
        https://arxiv.org/abs/1811.00894

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._moving_average import \
        MovingAverageSeriesTransformer
    >>> X = np.array([-3, -2, -1,  0,  1,  2,  3])
    >>> transformer = MovingAverageSeriesTransformer(2)
    >>> Xt = transformer.fit_transform(X)
    >>> print(Xt)
    [[-2.5 -1.5 -0.5  0.5  1.5  2.5]]
    """

    ...
