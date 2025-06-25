"""Exponential smoothing transformation."""

__maintainer__ = ["Datadote"]
__all__ = ["ExpSmoothingSeriesTransformer"]


from deprecated.sphinx import deprecated

from aeon.transformations.series.smoothing import ExponentialSmoothing


# TODO: Remove in v1.3.0
@deprecated(
    version="1.2.0",
    reason="ExpSmoothingSeriesTransformer is deprecated and will be removed in v1.3.0. "
    "Please use ExponentialSmoothing from "
    "transformations.series.smoothing instead.",
    category=FutureWarning,
)
class ExpSmoothingSeriesTransformer(ExponentialSmoothing):
    """Filter a time series using exponential smoothing.

    - Exponential smoothing (EXP) is a generalisaton of moving average smoothing that
    assigns a decaying weight to each element rather than averaging over a window.
    - Assume time series T = [t_0, ..., t_j], and smoothed values S = [s_0, ..., s_j]
    - Then, s_0 = t_0 and s_j = alpha * t_j + (1 - alpha) * s_j-1
    where 0 ≤ alpha ≤ 1. If window_size is given, alpha is overwritten, and set as
    alpha = 2. / (window_size + 1)

    Parameters
    ----------
    alpha: float, default=0.2
        decaying weight. Range [0, 1]. Overwritten by window_size if window_size exists
    window_size: int or float or None, default=None
        If window_size is specified, alpha is set to 2. / (window_size + 1)

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
    >>> from aeon.transformations.series._exp_smoothing import \
        ExpSmoothingSeriesTransformer
    >>> X = np.array([-2, -1,  0,  1,  2])
    >>> transformer = ExpSmoothingSeriesTransformer(0.5)
    >>> Xt = transformer.fit_transform(X)
    >>> print(Xt)
    [[-2.     -1.5    -0.75    0.125   1.0625]]
    >>> X = np.array([[1, 2, 3, 4], [10, 9, 8, 7]])
    >>> Xt = transformer.fit_transform(X)
    >>> print(Xt)
    [[ 1.     1.5    2.25   3.125]
     [10.     9.5    8.75   7.875]]
    """

    pass
