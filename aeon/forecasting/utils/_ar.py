r"""Estimators for pure autoregressive models.

Numba implementations of the sample autocovariance, the Levinson-Durbin
recursion, the Yule-Walker and Burg estimators and conditional least squares
for an AR(p) model

.. math::
    y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t.

The Yule-Walker and Burg estimators return the coefficients of every order up
to the maximum in one pass, which makes order selection cheap.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "autocovariance",
    "levinson_durbin",
    "yule_walker",
    "burg",
    "ar_ols",
    "ar_forecast",
]

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def autocovariance(y, max_lag):
    """Sample autocovariance of a series for lags ``0..max_lag``.

    Uses the standard (biased) estimator, with the mean of the whole series and
    divisor ``n`` at every lag. Unlike estimators with a separate mean or divisor
    per lag, this guarantees a positive semi-definite autocovariance matrix, so
    the Levinson-Durbin recursion on it gives a stationary AR model.

    Parameters
    ----------
    y : np.ndarray
        One dimensional time series of length ``n``.
    max_lag : int
        Largest lag, ``0 <= max_lag < n``.

    Returns
    -------
    np.ndarray
        Array of shape ``(max_lag + 1,)``, the autocovariance at lags
        ``0, 1, ..., max_lag``.
    """
    n = y.shape[0]
    x = y - np.mean(y)
    r = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        s = 0.0
        for t in range(k, n):
            s += x[t] * x[t - k]
        r[k] = s / n
    return r


@njit(cache=True)
def levinson_durbin(r, order):
    r"""Solve the Yule-Walker equations for every order up to ``order``.

    Parameters
    ----------
    r : np.ndarray
        Autocovariance at lags ``0..order`` or more.
    order : int
        Largest AR order.

    Returns
    -------
    coefs : np.ndarray
        Array of shape ``(order + 1, order)``. Row ``k`` holds the AR(k)
        coefficients :math:`\phi_1, \dots, \phi_k` in its first ``k`` entries,
        and zeros after them.
    sigma2 : np.ndarray
        Array of shape ``(order + 1,)``, the innovation variance of each order.
    pacf : np.ndarray
        Array of shape ``(order + 1,)``, the partial autocorrelations, with
        ``pacf[0] = 1``.
    """
    coefs = np.zeros((order + 1, order))
    sigma2 = np.zeros(order + 1)
    pacf = np.zeros(order + 1)
    pacf[0] = 1.0
    sigma2[0] = r[0]
    for k in range(1, order + 1):
        if sigma2[k - 1] <= 0.0:
            # Perfectly predictable at order k - 1: higher orders add nothing.
            for j in range(k - 1):
                coefs[k, j] = coefs[k - 1, j]
            continue
        acc = r[k]
        for j in range(1, k):
            acc -= coefs[k - 1, j - 1] * r[k - j]
        phi_kk = acc / sigma2[k - 1]
        for j in range(1, k):
            coefs[k, j - 1] = coefs[k - 1, j - 1] - phi_kk * coefs[k - 1, k - j - 1]
        coefs[k, k - 1] = phi_kk
        pacf[k] = phi_kk
        sigma2[k] = sigma2[k - 1] * (1.0 - phi_kk * phi_kk)
    return coefs, sigma2, pacf


@njit(cache=True)
def yule_walker(y, order):
    """Yule-Walker estimates of the AR coefficients for every order up to ``order``.

    The series is demeaned and the Levinson-Durbin recursion is run on its sample
    autocovariance (:func:`autocovariance`). This is the maximum likelihood form
    of the Yule-Walker estimator (divisor ``n`` at every lag), as in R's ``ar.yw``
    and ``statsmodels`` ``yule_walker(method="mle")``.

    Parameters
    ----------
    y : np.ndarray
        One dimensional time series of length ``n``.
    order : int
        Largest AR order, ``0 <= order < n``.

    Returns
    -------
    coefs, sigma2, pacf : np.ndarray
        As returned by :func:`levinson_durbin`.
    """
    return levinson_durbin(autocovariance(y, order), order)


@njit(cache=True, fastmath=True)
def burg(y, order):
    r"""Burg estimates of the AR coefficients for every order up to ``order``.

    Burg's method chooses each partial autocorrelation to minimise the sum of
    the squared forward and backward prediction errors. The fitted model is
    always stationary. The series is demeaned first. The coefficients match
    ``statsmodels`` ``burg`` and R's ``ar.burg``.

    The innovation variance is updated as
    :math:`\sigma^2_k = \sigma^2_{k-1}(1 - \kappa_k^2)` from
    :math:`\sigma^2_0 = \sum_t (y_t - \bar{y})^2 / n`, where :math:`\kappa_k`
    is the order k partial autocorrelation. This is R's default
    (``var.method = 1``). It keeps every order on the same scale, so the
    variances can be compared by AIC. ``statsmodels`` instead divides the order
    k error sum by ``n - k``, which favours larger orders.

    Parameters
    ----------
    y : np.ndarray
        One dimensional time series of length ``n``.
    order : int
        Largest AR order, ``0 <= order < n``.

    Returns
    -------
    coefs : np.ndarray
        Array of shape ``(order + 1, order)``. Row ``k`` holds the AR(k)
        coefficients in its first ``k`` entries.
    sigma2 : np.ndarray
        Array of shape ``(order + 1,)``, the innovation variance of each order.
    pacf : np.ndarray
        Array of shape ``(order + 1,)``, the partial autocorrelations (reflection
        coefficients), with ``pacf[0] = 1``.
    """
    n = y.shape[0]
    mean = np.mean(y)
    # After stage m, f[i] and b[i] hold the order m forward error at time
    # i + m and backward error at time i, for i < n - m.
    f = np.empty(n)  # forward prediction errors
    b = np.empty(n)  # backward prediction errors
    ss = 0.0
    for t in range(n):
        v = y[t] - mean
        f[t] = v
        b[t] = v
        ss += v * v
    coefs = np.zeros((order + 1, order))
    sigma2 = np.zeros(order + 1)
    pacf = np.zeros(order + 1)
    pacf[0] = 1.0
    sigma2[0] = ss / n
    length = n
    for m in range(1, order + 1):
        num = 0.0
        den = 0.0
        for i in range(length - 1):
            fi = f[i + 1]
            bi = b[i]
            num += fi * bi
            den += fi * fi + bi * bi
        k = 2.0 * num / den if den > 0.0 else 0.0
        pacf[m] = k
        for j in range(1, m):
            coefs[m, j - 1] = coefs[m - 1, j - 1] - k * coefs[m - 1, m - j - 1]
        coefs[m, m - 1] = k
        sigma2[m] = sigma2[m - 1] * (1.0 - k * k)
        length -= 1
        if m < order:
            for i in range(length):
                fi = f[i + 1]
                bi = b[i]
                f[i] = fi - k * bi
                b[i] = bi - k * fi
    return coefs, sigma2, pacf


@njit(cache=True, fastmath=True)
def ar_ols(y, order, start):
    r"""Least squares AR estimates, with intercept, for every order up to ``order``.

    For each ``k = 0..order``, regresses :math:`y_t` on
    :math:`1, y_{t-1}, \dots, y_{t-k}` for ``t = start, ..., n - 1``. This is the
    conditional sum of squares estimate. All orders use the same observations,
    so their residual sums of squares are comparable.

    The models are nested, so a single Cholesky factorisation of the
    cross-product matrix of ``[1, y_{t-1}, ..., y_{t-order}, y_t]`` gives the
    residual sum of squares and coefficients of every order. The series is
    centred first to improve the conditioning of the cross products. A
    regressor that is (numerically) a linear combination of earlier ones gets a
    zero coefficient.

    Parameters
    ----------
    y : np.ndarray
        One dimensional time series of length ``n``.
    order : int
        Largest AR order, ``order >= 0``.
    start : int
        First fitted time index, ``order <= start < n``.

    Returns
    -------
    coefs : np.ndarray
        Array of shape ``(order + 1, order)``. Row ``k`` holds the AR(k)
        coefficients in its first ``k`` entries.
    intercepts : np.ndarray
        Array of shape ``(order + 1,)``, the intercept :math:`c` of each order,
        on the scale of ``y``.
    rss : np.ndarray
        Array of shape ``(order + 1,)``, the residual sum of squares of each order
        over the ``n - start`` fitted observations.
    """
    n = y.shape[0]
    mean = np.mean(y)
    x = y - mean
    m = order + 2  # intercept, lags 1..order, then the target
    # G[a, b] = sum_t x[t - a] x[t - b] and T[a] = sum_t x[t - a] over
    # t = start..n-1, for lags a, b in 0..order. Lagging both series by one
    # moves the window back one step, so
    # G[a + 1, b + 1] = G[a, b] + x[start-1-a] x[start-1-b] - x[n-1-a] x[n-1-b],
    # and only the lag 0 row needs a full pass over the data.
    G = np.zeros((order + 1, order + 1))
    T = np.zeros(order + 1)
    s0 = 0.0
    for t in range(start, n):
        s0 += x[t]
    T[0] = s0
    for b in range(order + 1):
        acc = 0.0
        for t in range(start, n):
            acc += x[t] * x[t - b]
        G[0, b] = acc
    for a in range(order):
        T[a + 1] = T[a] + x[start - 1 - a] - x[n - 1 - a]
        for b in range(a, order):
            G[a + 1, b + 1] = (
                G[a, b]
                + x[start - 1 - a] * x[start - 1 - b]
                - x[n - 1 - a] * x[n - 1 - b]
            )
    # Cross products of [1, x[t-1], ..., x[t-order], x[t]] (upper triangle).
    A = np.zeros((m, m))
    A[0, 0] = n - start
    A[0, m - 1] = T[0]
    for i in range(1, m - 1):
        A[0, i] = T[i]
        A[i, m - 1] = G[0, i]
        for j in range(i, m - 1):
            A[i, j] = G[i, j]
    A[m - 1, m - 1] = G[0, 0]
    # Upper Cholesky factor R, with R.T @ R = A.
    R = np.zeros((m, m))
    for i in range(m):
        d = A[i, i]
        for k in range(i):
            d -= R[k, i] * R[k, i]
        if i < m - 1 and d <= 1e-12 * A[i, i]:
            continue  # redundant regressor: leave its row of R at zero
        if d < 0.0:
            d = 0.0
        R[i, i] = np.sqrt(d)
        if R[i, i] == 0.0:
            continue
        for j in range(i + 1, m):
            s = A[i, j]
            for k in range(i):
                s -= R[k, i] * R[k, j]
            R[i, j] = s / R[i, i]
    # RSS of the model with the first q regressors is the sum of the squared
    # entries of the target column below row q.
    last = m - 1
    rss = np.zeros(order + 1)
    tail = R[last, last] * R[last, last]
    for q in range(order + 1, 0, -1):
        rss[q - 1] = tail
        tail += R[q - 1, last] * R[q - 1, last]
    coefs = np.zeros((order + 1, order))
    intercepts = np.zeros(order + 1)
    beta = np.zeros(order + 1)
    for k in range(order + 1):
        # Solve R[:k+1, :k+1] beta = R[:k+1, last] by back-substitution.
        for i in range(k, -1, -1):
            if R[i, i] == 0.0:
                beta[i] = 0.0
                continue
            s = R[i, last]
            for j in range(i + 1, k + 1):
                s -= R[i, j] * beta[j]
            beta[i] = s / R[i, i]
        total = 0.0
        for j in range(k):
            coefs[k, j] = beta[j + 1]
            total += beta[j + 1]
        intercepts[k] = beta[0] + mean * (1.0 - total)
    return coefs, intercepts, rss


@njit(cache=True, fastmath=True)
def ar_forecast(y, intercept, coef, h):
    r"""Forecast ``h`` steps ahead from the end of ``y`` with an AR model.

    Parameters
    ----------
    y : np.ndarray
        One dimensional time series with at least ``len(coef)`` values.
    intercept : float
        The intercept :math:`c`.
    coef : np.ndarray
        The coefficients :math:`\phi_1, \dots, \phi_p`.
    h : int
        Number of steps to forecast, ``h >= 1``.

    Returns
    -------
    np.ndarray
        Array of shape ``(h,)``, the forecasts for steps ``1..h``.
    """
    p = coef.shape[0]
    n = y.shape[0]
    buf = np.empty(p + h)
    for j in range(p):
        buf[j] = y[n - p + j]
    for i in range(h):
        v = intercept
        for j in range(p):
            v += coef[j] * buf[p + i - 1 - j]
        buf[p + i] = v
    return buf[p:].copy()
