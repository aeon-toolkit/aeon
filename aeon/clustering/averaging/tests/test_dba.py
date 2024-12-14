"""Tests for DBA."""

import re

import numpy as np
import pytest

from aeon.clustering.averaging import (
    elastic_barycenter_average,
    petitjean_barycenter_average,
    subgradient_barycenter_average,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)

expected_petitjean_dba_univariate = np.array(
    [
        [
            1.4108696053220873,
            2.394226235189873,
            1.1060488942437015,
            2.623297561062815,
            0.527766384447058,
            1.43188098329184,
            1.6236753043608105,
            1.2058072160220672,
            1.1967764056706969,
            2.0004477079532927,
        ]
    ]
)

expected_petitjean_dba_multivariate = np.array(
    [
        [
            0.7267915954182468,
            1.3623320157069627,
            1.4495279596147028,
            0.5926686136095334,
            0.9971756690146295,
            1.3283383084721943,
            0.7659577046471742,
            1.9660527002664323,
            1.063255339395868,
            1.58993866039882,
        ],
        [
            1.2617459283113774,
            1.205815463685702,
            1.2234939638872127,
            1.7840697833529937,
            0.6722624075103683,
            1.835414289773766,
            0.9514286846149641,
            1.2908899681880017,
            1.416361929410518,
            1.5241706749949016,
        ],
        [
            1.3034322787744397,
            1.5436844012461954,
            1.632959525575443,
            1.0384357523605505,
            1.250310558432616,
            0.7533753079264169,
            0.7766566039307113,
            1.2180892807545585,
            1.6222668068442372,
            1.3287035370261573,
        ],
    ]
)

expected_subgradient_dba_univariate = np.array(
    [
        [
            1.51074199,
            2.15864417,
            1.00234603,
            2.35561396,
            0.63780835,
            1.40458666,
            1.51369944,
            1.2566171,
            1.18067802,
            1.8077327,
        ]
    ]
)

expected_subgradient_dba_multivariate = np.array(
    [
        [
            0.74762929,
            1.27990802,
            1.39635383,
            0.77782365,
            0.96456039,
            1.33559105,
            0.71501953,
            1.68703826,
            1.03765683,
            1.59958969,
        ],
        [
            1.28770927,
            1.23770584,
            1.20206139,
            1.78964445,
            0.89355146,
            1.6961715,
            1.00333335,
            1.23876289,
            1.38498497,
            1.55936623,
        ],
        [
            1.34461418,
            1.50480679,
            1.55114792,
            1.18503368,
            1.04103323,
            0.90804179,
            1.04520108,
            1.07537586,
            1.57586214,
            1.34801647,
        ],
    ]
)


def test_petitjean_dba():
    """Test petitjean dba functionality."""
    X_train_uni = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)

    average_ts_uni = elastic_barycenter_average(
        X_train_uni, method="petitjean", random_state=1
    )
    call_directly_average_ts_uni = petitjean_barycenter_average(
        X_train_uni, random_state=1
    )
    assert isinstance(average_ts_uni, np.ndarray)
    assert average_ts_uni.shape == X_train_uni[0].shape
    assert np.allclose(average_ts_uni, expected_petitjean_dba_univariate)
    assert np.allclose(average_ts_uni, call_directly_average_ts_uni)

    X_train_multi = make_example_3d_numpy(10, 3, 10, random_state=1, return_y=False)

    average_ts_multi = elastic_barycenter_average(
        X_train_multi, method="petitjean", random_state=1
    )
    call_directly_average_ts_multi = petitjean_barycenter_average(
        X_train_multi, random_state=1
    )

    assert isinstance(average_ts_multi, np.ndarray)
    assert average_ts_multi.shape == X_train_multi[0].shape
    assert np.allclose(average_ts_multi, expected_petitjean_dba_multivariate)
    assert np.allclose(average_ts_multi, call_directly_average_ts_multi)


def test_subgradient_dba():
    """Test stochastic subgradient dba functionality."""
    X_train_uni = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)

    average_ts_uni = elastic_barycenter_average(
        X_train_uni, method="subgradient", random_state=1
    )
    call_directly_average_ts_uni = subgradient_barycenter_average(
        X_train_uni, random_state=1
    )

    assert isinstance(average_ts_uni, np.ndarray)
    assert average_ts_uni.shape == X_train_uni[0].shape
    assert np.allclose(average_ts_uni, expected_subgradient_dba_univariate)
    assert np.allclose(average_ts_uni, call_directly_average_ts_uni)

    X_train_multi = make_example_3d_numpy(10, 3, 10, random_state=1, return_y=False)

    average_ts_multi = elastic_barycenter_average(
        X_train_multi, method="subgradient", random_state=1
    )
    call_directly_average_ts_multi = subgradient_barycenter_average(
        X_train_multi, random_state=1
    )

    assert isinstance(average_ts_multi, np.ndarray)
    assert average_ts_multi.shape == X_train_multi[0].shape
    assert np.allclose(average_ts_multi, expected_subgradient_dba_multivariate)
    assert np.allclose(average_ts_multi, call_directly_average_ts_multi)


@pytest.mark.parametrize(
    "distance",
    [
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "twe",
        "msm",
        "shape_dtw",
        "adtw",
    ],
)
def test_elastic_dba_variations(distance):
    """Test dba functionality with different distance measures."""
    X_train = make_example_3d_numpy(4, 2, 10, random_state=1, return_y=False)

    average_ts = elastic_barycenter_average(
        X_train, distance=distance, window=0.2, independent=False
    )

    subgradient_ts = elastic_barycenter_average(
        X_train, distance=distance, window=0.2, independent=False, method="subgradient"
    )

    assert isinstance(average_ts, np.ndarray)
    assert isinstance(subgradient_ts, np.ndarray)
    assert average_ts.shape == X_train[0].shape
    assert subgradient_ts.shape == X_train[0].shape


@pytest.mark.parametrize(
    "init_barycenter",
    [
        "mean",
        "medoids",
        "random",
        (
            make_example_1d_numpy(10, random_state=1),
            make_example_2d_numpy_series(n_timepoints=10, n_channels=4, random_state=1),
        ),
    ],
)
def test_dba_init(init_barycenter):
    """Test dba init functionality."""
    if isinstance(init_barycenter, str):
        univariate_init = init_barycenter
        multivariate_init = init_barycenter
    else:
        univariate_init = init_barycenter[0]
        multivariate_init = init_barycenter[1]

    X_train_uni = make_example_3d_numpy(4, 1, 10, random_state=1, return_y=False)
    average_ts_univ = elastic_barycenter_average(
        X_train_uni, window=0.2, init_barycenter=univariate_init
    )
    subgradient_ts_univ = elastic_barycenter_average(
        X_train_uni, window=0.2, method="subgradient", init_barycenter=univariate_init
    )

    assert isinstance(average_ts_univ, np.ndarray)
    assert isinstance(subgradient_ts_univ, np.ndarray)

    X_train_multi = make_example_3d_numpy(4, 4, 10, random_state=1, return_y=False)
    average_ts_multi = elastic_barycenter_average(
        X_train_multi, window=0.2, init_barycenter=multivariate_init
    )
    subgradient_ts_multi = elastic_barycenter_average(
        X_train_multi,
        window=0.2,
        method="subgradient",
        init_barycenter=multivariate_init,
    )

    assert isinstance(average_ts_multi, np.ndarray)
    assert isinstance(subgradient_ts_multi, np.ndarray)


def test_incorrect_input():
    """Test dba incorrect input."""
    # Test invalid distance
    X = make_example_3d_numpy(10, 1, 10, return_y=False)
    with pytest.raises(ValueError, match="Distance parameter invalid"):
        elastic_barycenter_average(X, distance="Distance parameter invalid")

    # Test invalid init barycenter string
    with pytest.raises(
        ValueError,
        match="init_barycenter string is invalid. Please use one of the "
        "following: 'mean', 'medoids', 'random'",
    ):
        elastic_barycenter_average(X, init_barycenter="init parameter invalid")

    # Test invalid init barycenter type
    with pytest.raises(
        ValueError,
        match="init_barycenter parameter is invalid. It must either be a "
        "str or a np.ndarray",
    ):
        elastic_barycenter_average(X, init_barycenter=[[1, 2, 3]])

    # Test invalid init barycenter with wrong shape
    with pytest.raises(
        ValueError,
        match=re.escape(
            "init_barycenter shape is invalid. Expected (1, 10) but " "got (1, 9)"
        ),
    ):
        elastic_barycenter_average(X, init_barycenter=make_example_1d_numpy(9))

    # Test invalid berycenter method
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid method: Not a real method. Please use one of the "
            "following: ['petitjean', 'subgradient']"
        ),
    ):
        elastic_barycenter_average(X, method="Not a real method")


def test_ba_weights():
    """Test weight parameter."""
    num_cases = 4
    X_train_uni = make_example_3d_numpy(
        num_cases, 1, 10, random_state=1, return_y=False
    )
    ones_weights = np.ones(num_cases)
    np.random.seed(1)
    random_weights = np.random.rand(num_cases)

    ba_no_weight_uni = elastic_barycenter_average(X_train_uni, random_state=1)
    ssg_ba_no_weight_uni = elastic_barycenter_average(
        X_train_uni, method="subgradient", random_state=1
    )

    ba_weights_ones_uni = elastic_barycenter_average(
        X_train_uni, weights=ones_weights, random_state=1
    )
    ssg_ba_weights_ones_uni = elastic_barycenter_average(
        X_train_uni, weights=ones_weights, method="subgradient", random_state=1
    )

    ba_weights_random_uni = elastic_barycenter_average(
        X_train_uni, weights=random_weights, random_state=1
    )
    ssg_ba_weights_random_uni = elastic_barycenter_average(
        X_train_uni, weights=random_weights, method="subgradient", random_state=1
    )

    assert np.array_equal(ba_no_weight_uni, ba_weights_ones_uni)
    assert np.array_equal(ssg_ba_no_weight_uni, ssg_ba_weights_ones_uni)
    assert not np.array_equal(ba_no_weight_uni, ba_weights_random_uni)
    assert not np.array_equal(ssg_ba_no_weight_uni, ssg_ba_weights_random_uni)

    X_train_multi = make_example_3d_numpy(
        num_cases, 4, 10, random_state=1, return_y=False
    )

    ba_no_wigtht_uni = elastic_barycenter_average(X_train_multi, random_state=1)
    ssg_ba_no_weight_multi = elastic_barycenter_average(
        X_train_multi, method="subgradient", random_state=1
    )
    ba_weights_ones_multi = elastic_barycenter_average(
        X_train_multi, weights=ones_weights, random_state=1
    )
    ssg_ba_weights_ones_multi = elastic_barycenter_average(
        X_train_multi, weights=ones_weights, method="subgradient", random_state=1
    )

    ba_weights_random_multi = elastic_barycenter_average(
        X_train_multi, weights=random_weights, random_state=1
    )
    ssg_ba_weights_random_multi = elastic_barycenter_average(
        X_train_multi, weights=random_weights, method="subgradient", random_state=1
    )

    assert np.array_equal(ba_no_wigtht_uni, ba_weights_ones_multi)
    assert np.array_equal(ssg_ba_no_weight_multi, ssg_ba_weights_ones_multi)
    assert not np.array_equal(ba_no_wigtht_uni, ba_weights_random_multi)
    assert not np.array_equal(ssg_ba_no_weight_multi, ssg_ba_weights_random_multi)
