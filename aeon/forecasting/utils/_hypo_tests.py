import numpy as np


def kpss_test(y, regression="c", lags=None):  # Test if time series is stationary
    """
    Perform the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test for stationarity.

    The KPSS test evaluates the null hypothesis that a time series is
    (trend or level) stationary against the alternative of a unit root
    (non-stationarity). It can test for either stationarity around a
    constant (level stationarity) or arounda deterministic trend
    (trend stationarity).

    Parameters
    ----------
    y : array-like
        Time series data to test for stationarity.
    regression : str, default="c"
        Indicates the null hypothesis for stationarity:
        - "c"  : Stationary around a constant (level stationarity)
        - "ct" : Stationary around a constant and linear trend (trend stationarity)
    lags : int or None, optional
        Number of lags to use for the
        HAC (heteroskedasticity and autocorrelation consistent) variance estimator.
        If None, defaults to sqrt(n), where n is the sample size.

    Returns
    -------
    kpss_stat : float
        The KPSS test statistic.
    stationary : bool
        True if the series is judged stationary at the 5% significance level
        (i.e., test statistic is below the critical value); False otherwise.

    Notes
    -----
    - Uses asymptotic 5% critical values from Kwiatkowski et al. (1992): 0.463 for level
      stationarity, 0.146 for trend stationarity.
    - Returns True for stationary if the test statistic is below the 5% critical value.

    References
    ----------
    Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992).
    "Testing the null hypothesis of stationarity against the alternative
    of a unit root."
    Journal of Econometrics, 54(1–3), 159–178.
    https://doi.org/10.1016/0304-4076(92)90104-Y

    Examples
    --------
    >>> from aeon.forecasting.utils._hypo_tests import kpss_test
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> kpss_test(y)
    (np.float64(1.1966313813502716), np.False_)
    """
    y = np.asarray(y)
    n = len(y)

    # Step 1: Fit regression model to estimate residuals
    if regression == "c":  # Constant
        X = np.ones((n, 1))
    elif regression == "ct":  # Constant + Trend
        X = np.column_stack((np.ones(n), np.arange(1, n + 1)))
    else:
        raise ValueError("regression must be 'c' or 'ct'")

    beta = np.linalg.lstsq(X, y, rcond=None)[0]  # Estimate regression coefficients
    residuals = y - X @ beta  # Get residuals (u_t)

    # Step 2: Compute cumulative sum of residuals (S_t)
    S_t = np.cumsum(residuals)

    # Step 3: Estimate long-run variance (HAC variance)
    if lags is None:
        # lags = int(12 * (n / 100)**(1/4)) # Default statsmodels lag length
        lags = int(np.sqrt(n))  # Default lag length

    gamma_0 = np.sum(residuals**2) / (n - X.shape[1])  # Lag-0 autocovariance
    gamma = [np.sum(residuals[k:] * residuals[:-k]) / n for k in range(1, lags + 1)]

    # Bartlett weights
    weights = [1 - (k / (lags + 1)) for k in range(1, lags + 1)]

    # Long-run variance
    sigma_squared = gamma_0 + 2 * np.sum([w * g for w, g in zip(weights, gamma)])

    # Step 4: Calculate the KPSS statistic
    kpss_stat = np.sum(S_t**2) / (n**2 * sigma_squared)

    # 5% critical values for KPSS test
    if regression == "ct":
        # p. 162 Kwiatkowski et al. (1992): y_t = beta * t + r_t + e_t,
        # where beta is the trend, r_t a random walk and e_t a stationary
        # error term.
        crit = 0.146
    else:  # hypo == "c"
        # special case of the model above, where beta = 0 (so the null
        # hypothesis is that the data is stationary around r_0).
        crit = 0.463

    return kpss_stat, kpss_stat < crit
