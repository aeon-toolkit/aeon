import random
import time

import numpy as np
from statsforecast.ets import etscalc
from statsforecast.utils import AirPassengers as ap

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
    start = time.time()
    f1 = ETSForecaster(alpha, beta, gamma, phi, 1, ModelType(error, trend, season, m))
    f1.fit(y)
    end = time.time()
    time_fitets = end - start
    e_fitets = f1.residuals_
    amse_fitets = f1.avg_mean_sq_err_
    lik_fitets = f1.liklihood_
    # Reinitialise arrays
    e.fill(0)
    amse.fill(0)
    f1 = ETSForecaster(alpha, beta, gamma, phi, 1, ModelType(error, trend, season, m))
    f1._initialise(y)
    init_states_etscalc = np.zeros(n * (1 + (trend > 0) + m * (season > 0) + 1))
    init_states_etscalc[0] = f1.level_
    init_states_etscalc[1] = f1.trend_
    init_states_etscalc[1 + (trend != 0) : m + 1 + (trend != 0)] = f1.season_[::-1]
    if season == 0:
        m = 1
    # Nixtla/statsforcast implementation
    start = time.time()
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
    end = time.time()
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
        print(time_fitets)  # noqa
        print(time_etscalc)  # noqa
        return True


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)
    # test_ets_comparison(setup, 241, False)
    SUCCESSES = True
    for i in range(0, 30000):
        SUCCESSES &= test_ets_comparison(setup, i, True)
    if SUCCESSES:
        print("Test Completed Successfully with no errors")  # noqa
