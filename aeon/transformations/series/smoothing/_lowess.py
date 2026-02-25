r"""LOWESS series smoother.

LOWESS (locally weighted scatterplot smoothing) is a local linear regression
smoother introduced by Cleveland (1979). This module implements a statsmodels-aligned
variant for equally spaced series, where the time index is assumed to be
:math:`t = 0, 1, \\ldots, n_{\\text{timepoints}} - 1`.

For each target time index :math:`i`:

1. Neighbourhood:
   Use :math:`k = \\mathrm{int}(\\mathrm{frac}\\,n + 10^{-10})` nearest indices in time
   (clamped to :math:`[2, n]`). On an equally spaced grid, this is a contiguous window
   around :math:`i`, adjusted at the boundaries.

2. Distance weights (tricube):
   Assign weights that decay with distance from :math:`i` using the tricube kernel:

   .. math::

      w(d) = \\left(1 - \\left(\\frac{d}{d_{\\max}}\\right)^3\\right)^3,

   where :math:`d_{\\max}` is the window radius.

3. Local fit (local linear regression):
   Fit a weighted least squares line over the neighbourhood:

   .. math::

      y_j \\approx a + b\\,(t_j - t_i),

   and take the fitted value at :math:`i` to be the local intercept :math:`a`.

Robust iterations (it)
----------------------
If :math:`\\texttt{it} > 0`, the smoother is made more resistant to outliers by
iteratively reweighting observations based on residuals. After an initial LOWESS fit,
residuals :math:`r_i = y_i - \\hat{y}_i` are computed and converted to robustness
weights using Tukey's bisquare function applied to

.. math::

   \\frac{|r_i|}{6\\,\\mathrm{median}(|r|)}.

The next LOWESS fit uses the product of the distance weights and these robustness
weights. This is repeated :math:`\\texttt{it}` times, so the total number of fits is
:math:`\\texttt{it} + 1`.

Notes
-----
- This implementation follows the behaviour of
   ``statsmodels.nonparametric.smoothers_lowess.lowess``  for the regular-grid case.
- It does not accept a separate exogenous :math:`x` array, and assumes an equally
   spaced grid.
- Missing values (NaN/inf) are not supported.

References
----------
Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots.
Journal of the American Statistical Association, 74(368), 829-836.

Cleveland, W. S., and Devlin, S. J. (1988). Locally weighted regression: an approach to
regression analysis by local fitting. Journal of the American Statistical Association,
83(403), 596-610.
"""

__maintainer__ = "TonyBagnall"
__all__ = ["LOWESS"]

import numpy as np
from numba import njit

from aeon.transformations.series.base import BaseSeriesTransformer


class LOWESS(BaseSeriesTransformer):
    r"""LOWESS smoother for equally spaced time series [1]_.

    This transformer applies LOWESS independently to each channel of an input
    series with shape ``(n_channels, n_timepoints)``. It implements the local
    linear LOWESS algorithm, plus optional robust reweighting iterations, in a
    way that is aligned with statsmodels' LOWESS for the regular-grid case.

    Parameters
    ----------
    frac : float, default=0.1
        Fraction of points used in each local regression window. The number of
        points used is
        :math:`k = \\mathrm{int}(\\mathrm{frac}\\,n_{\\text{timepoints}} + 10^{-10})`,
        clamped to :math:`[2, n_{\\text{timepoints}}]`. Set ``frac`` low for
        light smoothing.
    it : int, default=0
        Number of robust reweighting iterations. Total fits performed is
        :math:`\\texttt{it} + 1` (one initial fit plus :math:`\\texttt{it}`
        reweighted refits). Robust weights are computed from residuals using
        Tukey's bisquare function with scale
        :math:`6\\,\\mathrm{median}(|\\text{residual}|)`. Set higher if there may
        be outliers.
    delta : float, default=0.0
        A performance parameter for interpolation distance in :math:`x` units,
        matching the statsmodels concept. On an integer grid,
        :math:`\\delta < 1` typically has no effect. Larger values can reduce
        computation by skipping some fits and linearly interpolating between
        fitted points.

    Notes
    -----
    - Regular grid only, the time index is assumed to be
      :math:`0, \\ldots, n_{\\text{timepoints}} - 1`.
    - Missing values are not supported.

    References
    ----------
    .. [1] Cleveland, W. S. (1979). Robust locally weighted regression and
           smoothing scatterplots. Journal of the American Statistical
           Association, 74(368), 829-836.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series.smoothing import LOWESS
    >>> X = np.random.RandomState(0).randn(1, 20)  # (n_channels, n_timepoints)
    >>> transformer = LOWESS(frac=0.3, it=0)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (1, 20)
    """

    _tags = {
        "fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(self, frac: float = 0.1, it: int = 0, delta: float = 0.0):
        self.frac = frac
        self.it = it
        self.delta = delta
        super().__init__(axis=1)

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        # Numba safety
        frac = float(self.frac)
        it = int(self.it)
        delta = float(self.delta)

        if not (0.0 <= frac <= 1.0):
            raise ValueError("frac must be in [0, 1].")
        if it < 0:
            raise ValueError("it must be >= 0.")
        if delta < 0.0:
            raise ValueError("delta must be >= 0.")

        X2 = X.astype(np.float64, copy=False)
        return _lowess_2d(X2, frac, it, delta)


@njit(cache=True, fastmath=True)
def _tricube_from_distnorm(dn: float) -> float:
    """Tricube operator: (1 - dn^3)^3, dn in [0, 1]."""
    t2 = dn * dn
    t3 = t2 * dn
    u = 1.0 - t3
    return u * u * u


@njit(cache=True, fastmath=True)
def _bisquare(z: float) -> float:
    """Bisquare operator: (1 - z^2)^2, with z in [0, 1]."""
    tmp = 1.0 - z * z
    return tmp * tmp


@njit(cache=True)
def _median_1d(a: np.ndarray) -> float:
    """Median matching numpy median (average of the two middle values for even n).

    Could be made marginally faster using partition rather than sort.
    """
    b = np.sort(a.copy())
    n = b.size
    m = n // 2
    if n % 2 == 1:
        return float(b[m])
    return 0.5 * (float(b[m - 1]) + float(b[m]))


@njit(cache=True, fastmath=True)
def _calc_resid_weights(y: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """Residual weights: bisquare(|r| / (6 * median(|r|))), clipped to 1."""
    n = y.size
    abs_resid = np.empty(n, dtype=np.float64)
    for i in range(n):
        r = y[i] - y_fit[i]
        abs_resid[i] = r if r >= 0.0 else -r

    med = _median_1d(abs_resid)

    z = np.empty(n, dtype=np.float64)
    if med == 0.0:
        # statsmodels: z = (abs_resid > 0) then bisquare(z)
        for i in range(n):
            z[i] = 1.0 if abs_resid[i] > 0.0 else 0.0
    else:
        scale = 6.0 * med
        for i in range(n):
            zi = abs_resid[i] / scale
            if zi > 1.0:
                zi = 1.0
            z[i] = zi

    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        w[i] = _bisquare(z[i])
    return w


@njit(cache=True, fastmath=True)
def _fit_one_point_regular(
    y: np.ndarray,
    resid_w: np.ndarray,
    i: int,
    k: int,
) -> float:
    """Compute the LOWESS fitted value at index i on a regular grid x=j."""
    n = y.size

    # Regular-grid neighbourhood equivalent to statsmodels update_neighborhood
    left = i - (k // 2)
    if left < 0:
        left = 0
    right = left + k
    if right > n:
        right = n
        left = n - k
        if left < 0:
            left = 0

    # Radius as in statsmodels: max distance to leftmost/rightmost neighbour
    dl = i - left
    dr = (right - 1) - i
    radius = dl if dl > dr else dr
    if radius <= 0:
        # Degenerate: only possible for n=1
        return y[i]

    # Compute unnormalised weights
    sum_w = 0.0
    nonzero = 0
    wloc = np.empty(right - left, dtype=np.float64)

    for t, j in enumerate(range(left, right)):
        d = j - i
        if d < 0:
            d = -d
        dn = float(d) / float(radius)  # in [0, 1]
        w = _tricube_from_distnorm(dn) * resid_w[j]
        wloc[t] = w
        sum_w += w
        if w > 1e-12:
            nonzero += 1

    # statsmodels requires at least 2 non-zero weights for a valid regression
    if nonzero < 2 or sum_w <= 0.0:
        return y[i]

    # Normalise weights to sum to 1
    inv_sum = 1.0 / sum_w
    for t in range(wloc.size):
        wloc[t] *= inv_sum

    # Projection-form local linear regression (matches statsmodels calculate_y_fit)
    sum_wx = 0.0
    for t, j in enumerate(range(left, right)):
        sum_wx += wloc[t] * float(j)

    sq = 0.0
    for t, j in enumerate(range(left, right)):
        dx = float(j) - sum_wx
        sq += wloc[t] * dx * dx
    if sq < 1e-12:
        sq = 1e-12

    yi = 0.0
    for t, j in enumerate(range(left, right)):
        dxj = float(j) - sum_wx
        p = wloc[t] * (1.0 + (float(i) - sum_wx) * dxj / sq)
        yi += p * y[j]

    return yi


@njit(cache=True, fastmath=True)
def _interpolate_segment(y_fit: np.ndarray, left_i: int, right_i: int) -> None:
    """Linearly interpolate y_fit between fitted points at left_i and right_i."""
    denom = float(right_i - left_i)
    if denom <= 0.0:
        return
    y0 = y_fit[left_i]
    y1 = y_fit[right_i]
    for j in range(left_i + 1, right_i):
        a = float(j - left_i) / denom
        y_fit[j] = a * y1 + (1.0 - a) * y0


@njit(cache=True, fastmath=True)
def _lowess_1d(y: np.ndarray, frac: float, it: int, delta: float) -> np.ndarray:
    """LOWESS for 1D y on a regular grid x=0..n-1."""
    n = y.size

    # statsmodels neighbour count: k = int(frac*n + 1e-10), clamped to [2, n]
    k = int(frac * n + 1e-10)
    if k < 2:
        k = 2
    if k > n:
        k = n

    # statsmodels semantics: it is number of robust reweightings, plus 1 initial fit
    total_iters = it + 1

    resid_w = np.ones(n, dtype=np.float64)
    y_fit = np.empty(n, dtype=np.float64)

    # delta skipping only has effect if delta >= 1 on an integer grid
    for _ in range(total_iters):
        for j in range(n):
            y_fit[j] = 0.0

        i = 0
        last_fit_i = -1

        while True:
            y_fit[i] = _fit_one_point_regular(y, resid_w, i, k)

            if last_fit_i >= 0 and last_fit_i < i - 1:
                _interpolate_segment(y_fit, last_fit_i, i)

            last_fit_i = i
            if last_fit_i >= n - 1:
                break

            if delta <= 0.0:
                i = last_fit_i + 1
                continue

            cutpoint = float(last_fit_i) + delta
            nxt = last_fit_i + 1
            while nxt < n and float(nxt) <= cutpoint:
                nxt += 1

            cand = nxt - 1
            if cand < last_fit_i + 1:
                cand = last_fit_i + 1
            if cand >= n:
                cand = n - 1
            i = cand

        resid_w = _calc_resid_weights(y, y_fit)

    return y_fit


@njit(cache=True, fastmath=True)
def _lowess_2d(X: np.ndarray, frac: float, it: int, delta: float) -> np.ndarray:
    """Apply 1D LOWESS independently per channel."""
    n_channels, n = X.shape
    out = np.empty((n_channels, n), dtype=np.float64)
    for c in range(n_channels):
        out[c, :] = _lowess_1d(X[c, :], frac, it, delta)
    return out
