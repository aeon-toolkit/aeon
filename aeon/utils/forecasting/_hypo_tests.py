import numpy as np


def kpss_test(y, regression="c", lags=None):  # Test if time series is stationary
    """
    Implement the KPSS test for stationarity.

    Parameters
    ----------
    y (array-like): Time series data
    regression (str): 'c' for constant, 'ct' for constant + trend
    lags (int): Number of lags for HAC variance estimation (default: sqrt(n))

    Returns
    -------
    kpss_stat (float): KPSS test statistic
    stationary (bool): Whether the series is stationary according to the test
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
