"""Script to test ETS implementation against ETS implementations from other modules."""

import random
import time
import timeit

import numpy as np
from statsforecast.utils import AirPassengers as ap

import aeon.forecasting._ets as ets
from aeon.forecasting import ETSForecaster

NA = -99999.0
MAX_NMSE = 30
MAX_SEASONAL_PERIOD = 24


def setup():
    """Generate parameters required for ETS algorithms."""
    y = ap
    m = random.randint(2, 24)
    error = random.randint(1, 2)
    trend = random.randint(0, 2)
    season = random.randint(0, 2)
    alpha = round(random.random(), 4)
    if alpha == 0:
        alpha = round(random.random(), 4)
    beta = round(random.random() * alpha, 4)  # 0 < beta < alpha
    if beta == 0:
        beta = round(random.random() * alpha, 4)
    gamma = round(random.random() * (1 - alpha), 4)  # 0 < beta < alpha
    if gamma == 0:
        gamma = round(random.random() * (1 - alpha), 4)
    phi = round(
        random.random() * 0.18 + 0.8, 4
    )  # Common constraint for phi is 0.8 < phi < 0.98
    return (y, m, error, trend, season, alpha, beta, gamma, phi)


def statsmodels_version(
    y: np.ndarray,
    m: int,
    f1: ETSForecaster,
    errortype: int,
    trendtype: int,
    seasontype: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
):
    """Hide the differences between different statsforecast versions."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ets_model = ExponentialSmoothing(
        y[m:],
        trend="add" if trendtype == 1 else "mul" if trendtype == 2 else None,
        damped_trend=(phi != 1 and trendtype != 0),
        seasonal="add" if seasontype == 1 else "mul" if seasontype == 2 else None,
        seasonal_periods=m if seasontype != 0 else None,
        initialization_method="known",
        initial_level=f1.level_,
        initial_trend=f1.trend_ if trendtype != 0 else None,
        initial_seasonal=f1.seasonality_ if seasontype != 0 else None,
    )
    results = ets_model.fit(
        smoothing_level=alpha,
        smoothing_trend=(
            beta / alpha if trendtype != 0 else None
        ),  # statsmodels uses beta*=beta/alpha
        smoothing_seasonal=gamma if seasontype != 0 else None,
        damping_trend=phi if trendtype != 0 else None,
        optimized=False,
    )
    avg_mean_sq_err = results.sse / (len(y) - m)
    # Back-calculate our log-likelihood proxy from AIC
    if errortype == 1:
        residuals = y[m:] - results.fittedvalues
        assert np.allclose(residuals, results.resid)
    else:
        residuals = y[m:] / results.fittedvalues - 1
    return (
        (np.array([avg_mean_sq_err])),
        residuals,
        (results.aic - 2 * results.k + (len(y) - m) * np.log(len(y) - m)),
    )


def obscure_statsforecast_version(
    y: np.ndarray,
    m: int,
    f1: ETSForecaster,
    errortype: int,
    trendtype: int,
    seasontype: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
):
    """Hide the differences between different statsforecast versions."""
    init_state = np.zeros(len(y) * (1 + (trendtype > 0) + m * (seasontype > 0) + 1))
    init_state[0] = f1.level_
    init_state[1] = f1.trend_
    init_state[1 + (trendtype != 0) : m + 1 + (trendtype != 0)] = f1.seasonality_[::-1]
    # from statsforecast.ets import pegelsresid_C
    # amse, e, _x, lik = pegelsresid_C(
    #     y[m:],
    #     m,
    #     init_state,
    #     "A" if errortype == 1 else "M",
    #     "A" if trendtype == 1 else "M" if trendtype == 2 else "N",
    #     "A" if seasontype == 1 else "M" if seasontype == 2 else "N",
    #     phi != 1,
    #     alpha,
    #     beta,
    #     gamma,
    #     phi,
    #     nmse,
    # )
    from statsforecast.ets import etscalc

    e = np.zeros(len(y))
    amse = np.zeros(MAX_NMSE)
    lik = etscalc(
        y[m:],
        len(y) - m,
        init_state,
        m,
        errortype,
        trendtype,
        seasontype,
        alpha,
        beta,
        gamma,
        phi,
        e,
        amse,
        1,
    )
    return amse, e[:-m], lik


def test_ets_comparison(setup_func, random_seed, catch_errors):
    """Run both our statsforecast and our implementation and crosschecks."""
    random.seed(random_seed)
    (
        y,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
    ) = setup_func()
    # tsml-eval implementation
    start = time.perf_counter()
    f1 = ETSForecaster(
        error,
        trend,
        season,
        m,
        alpha,
        beta,
        gamma,
        phi,
        1,
    )
    f1.fit(y)
    end = time.perf_counter()
    time_fitets = end - start
    e_fitets = f1.residuals_
    amse_fitets = f1.avg_mean_sq_err_
    lik_fitets = f1.liklihood_
    f1 = ETSForecaster(error, trend, season, m, alpha, beta, gamma, phi, 1)
    # pylint: disable=W0212
    f1._fit(y)._initialise(y)
    # pylint: enable=W0212
    if season == 0:
        m = 1
    # Nixtla/statsforcast implementation
    start = time.perf_counter()
    amse_etscalc, e_etscalc, lik_etscalc = statsmodels_version(
        y, m, f1, error, trend, season, alpha, beta, gamma, phi
    )
    end = time.perf_counter()
    time_etscalc = end - start
    amse_etscalc = amse_etscalc[0]

    if catch_errors:
        try:
            # Comparing outputs and runtime
            assert np.allclose(e_fitets, e_etscalc), "Residuals Compare failed"
            assert np.allclose(amse_fitets, amse_etscalc), "AMSE Compare failed"
            assert np.isclose(lik_fitets, lik_etscalc), "Liklihood Compare failed"
            return True
        except AssertionError as e:
            print(e)  # noqa
            print(  # noqa
                f"Seed: {random_seed}, Model: Error={error}, Trend={trend},\
                   Seasonality={season}, seasonal period={m},\
                   alpha={alpha}, beta={beta}, gamma={gamma}, phi={phi}"
            )
            return False
    else:
        print(  # noqa
            f"Seed: {random_seed}, Model: Error={error}, Trend={trend},\
                Seasonality={season}, seasonal period={m}, alpha={alpha},\
                    beta={beta}, gamma={gamma}, phi={phi}"
        )
        diff_indices = np.where(
            np.abs(e_fitets - e_etscalc) > 1e-3 * np.abs(e_etscalc) + 1e-2
        )[0]
        for index in diff_indices:
            print(  # noqa
                f"Index {index}: e_fitets = {e_fitets[index]},\
                e_etscalc = {e_etscalc[index]}"
            )
        print(amse_fitets)  # noqa
        print(amse_etscalc)  # noqa
        print(lik_fitets)  # noqa
        print(lik_etscalc)  # noqa
        assert np.allclose(e_fitets, e_etscalc)
        assert np.allclose(amse_fitets, amse_etscalc)
        # assert np.isclose(lik_fitets, lik_etscalc)
        print(f"Time for ETS: {time_fitets:0.20f}")  # noqa
        print(f"Time for statsforecast ETS: {time_etscalc}")  # noqa
        return True


def time_ets():
    """Test function for optimised numba ets algorithm."""
    ETSForecaster(2, 2, 2, 4).fit(ap).predict()


def time_sf():
    """Test function for statsforecast ets algorithm."""
    x = np.zeros(144 * 7)
    x[0:6] = [122.75, 1.123230970596215, 0.91242363, 0.96130346, 1.07535642, 1.0509165]
    obscure_statsforecast_version(
        ap[4:],
        4,
        x,
        2,
        2,
        2,
        0.1,
        0.01,
        0.01,
        0.99,
    )


def time_compare(random_seed):
    """Compare timings of different ets algorithms."""
    random.seed(random_seed)
    (y, m, error, trend, season, alpha, beta, gamma, phi) = setup()
    # etsnoopt_time = timeit.timeit(time_etsnoopt, globals={}, number=10000)
    # print (f"Execution time ETS No-opt: {etsnoopt_time} seconds")
    # Do a few iterations to remove background/overheads. Makes comparison more reliable
    for _i in range(10):
        time_ets()
        time_sf()
    ets_time = timeit.timeit(time_ets, globals={}, number=1000)
    print(f"Execution time ETS: {ets_time} seconds")  # noqa
    statsforecast_time = timeit.timeit(time_sf, globals={}, number=1000)
    print(f"Execution time StatsForecast: {statsforecast_time} seconds")  # noqa
    ets_time = timeit.timeit(time_ets, globals={}, number=1000)
    print(f"Execution time ETS: {ets_time} seconds")  # noqa
    statsforecast_time = timeit.timeit(time_sf, globals={}, number=1000)
    print(f"Execution time StatsForecast: {statsforecast_time} seconds")  # noqa
    # _ets implementation
    start = time.perf_counter()
    f1 = ets.ETSForecaster(error, trend, season, m, alpha, beta, gamma, phi, 1)
    f1.fit(y)
    end = time.perf_counter()
    ets_time = end - start
    print(f"ETS Time: {ets_time}")  # noqa
    return ets_time


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    test_ets_comparison(setup, 300, False)
    SUCCESSES = True
    for i in range(0, 300):
        SUCCESSES &= test_ets_comparison(setup, i, True)
    if SUCCESSES:
        print("Test Completed Successfully with no errors")  # noqa
    # time_compare(300)
    # avg_ets = 0
    # iterations = 100
    # for i in range (iterations):
    #     time_ets = time_compare(300)
    #     avg_ets += time_ets
    # avg_ets/= iterations
    # print(f"Avg ETS Time: {avg_ets},\
