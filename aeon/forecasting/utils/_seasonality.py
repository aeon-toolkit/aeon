"""Seasonality tests."""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def calc_seasonal_period(data):
    """Calculate the seasonal period of a time series."""
    n = len(data)
    
    # --- QUANT GUARD 1: Series too short ---
    # You cannot have seasonality if you don't even have 4 points.
    if n < 4:
        return 1
    # ---------------------------------------

    # Limit the max seasonal period we check to half the data length.
    # This prevents the "lag_length = 0" crash.
    max_lag = int(n / 2)
    if max_lag < 2:
        return 1

    # Calculate mean manually for Numba efficiency
    mean = 0.0
    for x in data:
        mean += x
    mean /= n

    # Calculate variance (denominator for ACF)
    denom = 0.0
    for x in data:
        diff = x - mean
        denom += diff * diff
    
    # --- QUANT GUARD 2: Zero Variance ---
    # If the line is flat (variance ~ 0), it has no seasonality.
    if denom < 1e-10:
        return 1
    # ------------------------------------

    # Find the lag with the highest autocorrelation
    best_acf = -1.0
    best_lag = 1

    for lag in range(2, max_lag + 1):
        num = 0.0
        # Calculate autocovariance for this lag
        for i in range(n - lag):
            num += (data[i] - mean) * (data[i + lag] - mean)
        
        acf = num / denom
        if acf > best_acf:
            best_acf = acf
            best_lag = lag

    # A simple heuristic: if the best ACF is very weak, assume no seasonality
    if best_acf < 0.2:
        return 1
        
    return best_lag
