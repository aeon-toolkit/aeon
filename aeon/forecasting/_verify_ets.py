import random
import time
import timeit

import numpy as np
from statsforecast.ets import etscalc
from statsforecast.utils import AirPassengers as ap

import aeon.forecasting._ets_fast as etsfast
import aeon.forecasting._ets_fast_structtest as ets_structtest
from aeon.forecasting import ETSForecaster, ModelType

NA = -99999.0
MAX_NMSE = 30
MAX_SEASONAL_PERIOD = 24


def setup():
    """Generate parameters required for ETS algorithms."""
    y = ap
    n = len(ap)
    m = random.randint(1, 24)
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
    e = np.zeros(n)
    lik_fitets = np.zeros(1)
    amse = np.zeros(MAX_NMSE)
    nmse = 3
    return (
        y,
        n,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        e,
        lik_fitets,
        amse,
        nmse,
    )


def test_ets_comparison(setup_func, random_seed, catch_errors):
    """Run both our statsforecast and our implementation and crosschecks."""
    random.seed(random_seed)
    (
        y,
        n,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        e,
        lik_fitets,
        amse,
        nmse,
    ) = setup_func()
    # tsml-eval implementation
    start = time.perf_counter()
    f1 = ETSForecaster(
        ModelType(error, trend, season, m),
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
    # Reinitialise arrays
    e.fill(0)
    amse.fill(0)
    f1 = ETSForecaster(ModelType(error, trend, season, m), alpha, beta, gamma, phi, 1)
    f1._initialise(y)
    init_states_etscalc = np.zeros(n * (1 + (trend > 0) + m * (season > 0) + 1))
    init_states_etscalc[0] = f1.level_
    init_states_etscalc[1] = f1.trend_
    init_states_etscalc[1 + (trend != 0) : m + 1 + (trend != 0)] = f1.season_[::-1]
    if season == 0:
        m = 1
    # Nixtla/statsforcast implementation
    start = time.perf_counter()
    lik_etscalc = etscalc(
        y[m:],
        n - m,
        init_states_etscalc,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        e,
        amse,
        nmse,
    )
    end = time.perf_counter()
    time_etscalc = end - start
    e_etscalc = e.copy()
    amse_etscalc = amse.copy()[0]

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
            np.abs(e_fitets - e_etscalc) > 1e-5 * np.abs(e_etscalc) + 1e-8
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
        # assert np.allclose(init_states_fitets, init_states_etscalc)
        assert np.allclose(e_fitets, e_etscalc)
        assert np.allclose(amse_fitets, amse_etscalc)
        assert np.isclose(lik_fitets, lik_etscalc)
        print(f"Time for ETS: {time_fitets:0.20f}")  # noqa
        print(f"Time for statsforecast ETS: {time_etscalc}")  # noqa
        return True


def time_etsfast():
    """Test function for optimised numba ets algorithm."""
    etsfast.ETSForecaster(2, 2, 2, 4).fit(ap).predict()


def time_ets_structtest():
    """Test function for ets algorithm using classes."""
    ets_structtest.ETSForecaster(ets_structtest.ModelType(2, 2, 2, 4)).fit(ap).predict()


def time_etsnoopt():
    """Test function for non-optimised ets algorithm."""
    ETSForecaster(ModelType(2, 2, 2, 4)).fit(ap).predict()


def time_etsfast_noclass():
    """Test function for optimised ets algorithm without the class based structure."""
    data = np.array(ap.squeeze(), dtype=np.float64)
    (level, trend, seasonality, residuals_, avg_mean_sq_err_, liklihood_) = (
        etsfast._fit(data, 2, 2, 2, 4, 0.1, 0.01, 0.01, 0.99)
    )
    etsfast._predict(2, 2, level, trend, seasonality, 0.99, 1, 144, 4)


def time_sf():
    """Test function for statsforecast ets algorithm."""
    x = np.zeros(144 * 7)
    x[0:6] = [122.75, 1.123230970596215, 0.91242363, 0.96130346, 1.07535642, 1.0509165]
    etscalc(
        ap[4:],
        140,
        x,
        4,
        2,
        2,
        2,
        0.1,
        0.01,
        0.01,
        0.99,
        np.zeros(144),
        np.zeros(30),
        1,
    )


def time_compare(random_seed):
    """Compare timings of different ets algorithms."""
    random.seed(random_seed)
    (
        y,
        n,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        e,
        lik_fitets,
        amse,
        nmse,
    ) = setup()
    # etsnoopt_time = timeit.timeit(time_etsnoopt, globals={}, number=10000)
    # print (f"Execution time ETS No-opt: {etsnoopt_time} seconds")
    # ets_structtest_time = timeit.timeit(time_ets_structtest, globals={}, number=10000)
    # print (f"Execution time ETS Structtest: {ets_structtest_time} seconds")
    for _i in range(10):
        time_etsfast()
        time_sf()
        time_etsfast_noclass()
    etsfast_time = timeit.timeit(time_etsfast, globals={}, number=1000)
    print(f"Execution time ETS Fast: {etsfast_time} seconds")  # noqa
    etsfast_noclass_time = timeit.timeit(time_etsfast_noclass, globals={}, number=1000)
    print(f"Execution time ETS Fast NoClass: {etsfast_noclass_time} seconds")  # noqa
    statsforecast_time = timeit.timeit(time_sf, globals={}, number=1000)
    print(f"Execution time StatsForecast: {statsforecast_time} seconds")  # noqa
    etsfast_time = timeit.timeit(time_etsfast, globals={}, number=1000)
    print(f"Execution time ETS Fast: {etsfast_time} seconds")  # noqa
    etsfast_noclass_time = timeit.timeit(time_etsfast_noclass, globals={}, number=1000)
    print(f"Execution time ETS Fast NoClass: {etsfast_noclass_time} seconds")  # noqa
    statsforecast_time = timeit.timeit(time_sf, globals={}, number=1000)
    print(f"Execution time StatsForecast: {statsforecast_time} seconds")  # noqa
    # _ets_fast_nostruct implementation
    start = time.perf_counter()
    f3 = etsfast.ETSForecaster(error, trend, season, m, alpha, beta, gamma, phi, 1)
    f3.fit(y)
    end = time.perf_counter()
    etsfast_time = end - start
    # _ets_fast implementation
    start = time.perf_counter()
    f2 = ets_structtest.ETSForecaster(
        ets_structtest.ModelType(error, trend, season, m),
        ets_structtest.SmoothingParameters(alpha, beta, gamma, phi),
        1,
    )
    f2.fit(y)
    end = time.perf_counter()
    ets_structtest_time = end - start
    # _ets implementation
    start = time.perf_counter()
    f1 = ETSForecaster(ModelType(error, trend, season, m), alpha, beta, gamma, phi, 1)
    f1.fit(y)
    end = time.perf_counter()
    etsnoopt_time = end - start
    assert np.allclose(f1.residuals_, f2.residuals_)
    assert np.allclose(f1.avg_mean_sq_err_, f2.avg_mean_sq_err_)
    assert np.isclose(f1.liklihood_, f2.liklihood_)
    assert np.allclose(f1.residuals_, f3.residuals_)
    assert np.allclose(f1.avg_mean_sq_err_, f3.avg_mean_sq_err_)
    assert np.isclose(f1.liklihood_, f3.liklihood_)
    print(  # noqa
        f"ETS No-optimisation Time: {etsnoopt_time},\
        Fast Structtest time: {ets_structtest_time},\
        Fast time: {etsfast_time}"
    )
    return etsnoopt_time, ets_structtest_time, etsfast_time


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)
    # test_ets_comparison(setup, 300, False)
    # SUCCESSES = True
    # for i in range(0, 30000):
    #     SUCCESSES &= test_ets_comparison(setup, i, True)
    # if SUCCESSES:
    #     print("Test Completed Successfully with no errors")  # noqa
    time_compare(300)
    # avg_ets = 0
    # avg_etsfast = 0
    # avg_etsfast_ns = 0
    # iterations = 100
    # for i in range (iterations):
    #     time_ets, ets_structtest_time, etsfast_time = time_compare(300)
    #     avg_ets += time_ets
    #     avg_etsfast += time_etsfast
    #     avg_etsfast_ns += time_etsfast_nostruct
    # avg_ets/= iterations
    # avg_etsfast/= iterations
    # avg_etsfast_ns /= iterations
    # print(f"Avg ETS Time: {avg_ets}, Avg Fast ETS time: {avg_etsfast},\
    # Avg Fast Nostruct time: {avg_etsfast_ns}")
