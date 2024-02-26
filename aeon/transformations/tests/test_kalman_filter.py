"""Kalman Filter transformers unit tests."""

__maintainer__ = []

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.transformations.kalman_filter import KalmanFilterTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies

# ts stands for time steps
ts = 10


def create_data(shape, missing_values=False, p=0.15, mult=10):
    """Create random ndarray of shape `shape`.

    The result array will contain missing values (represented by np.nan)
    if parameter `missing_values` is set to true.
    """
    if isinstance(shape, int):
        shape = (shape,)
    data = np.random.rand(*shape) * mult

    if missing_values:
        time_steps = shape[0]
        measurement_dim = shape[1]
        pr = max(int(time_steps * p), 4)
        for t in range(time_steps):
            if t % pr == 0:
                data[t] = [np.nan] * measurement_dim
    return data


def rand_list(shape, length=ts, mult=10):
    """Return list with `length` random ndarrays of shape `shape`."""
    return [create_data(shape, mult=mult) for _ in range(length)]


# state_dim = 3, measurement_dim = 3, time_steps = ts (10)
params_3_3_dynamic = {
    "process_noise": create_data((ts, 3, 3)),  # random ndarray of shape (ts, 3, 3),
    "measurement_noise": rand_list(
        (3, 3), length=ts
    ),  # [`ts` random ndarrays of shape (3, 3)],
    "measurement_function": create_data((3, 3)),  # random ndarray of shape (3, 3),
    "initial_state_covariance": create_data((3, 3)),  # random ndarray of shape (3, 3),
}

# state_dim = 3, measurement_dim = 3, time_steps = ts (10)
params_3_3_static = {
    "state_transition": create_data((3, 3)),  # random ndarray of shape (3,3),
    "process_noise": create_data((3, 3)),  # random ndarray of shape (3,3),
    "measurement_noise": create_data((3, 3)),  # random ndarray of shape (3,3),
    "initial_state": create_data(3),  # random ndarray of shape (3,),
    "initial_state_covariance": create_data((3, 3)),  # random ndarray of shape (3,3)
}

# state_dim = 2, measurement_dim = 3, time_steps = ts (10)
params_2_3_ = {
    "state_transition": rand_list(
        (2, 2), length=ts
    ),  # [`ts` random ndarrays of shape (2,2)],
    "measurement_noise": create_data((3, 3)),  # random ndarray of shape (3, 3),
    "measurement_function": rand_list(
        (3, 2), length=ts
    ),  # [`ts` random ndarrays of shape (3,2)],
    "initial_state": create_data(2),  # random ndarray of shape (2,),
    "initial_state_covariance": create_data((2, 2)),  # random ndarray of shape (2,2)
}

# state_dim = 1, measurement_dim = 1, time_steps = ts (10)
params_1_1_arrays = {
    "state_transition": create_data((ts, 1, 1)),  # random ndarray of shape (ts, 1, 1)
    "process_noise": create_data((ts, 1, 1)),  # random ndarray of shape (ts, 1, 1)
    "initial_state": create_data(1),  # random ndarray of shape (1,)
    "initial_state_covariance": create_data((1, 1)),  # random ndarray of shape (1, 1)
}

# state_dim = 1, measurement_dim = 1, time_steps = ts (10)
params_1_1_lists = {
    "process_noise": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "measurement_noise": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "measurement_function": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "initial_state": create_data(1),  # random ndarray of shape (1,)
}

# state_dim = 3, measurement_dim = 1, time_steps = ts (10)
params_3_1_lists = {
    "state_transition": rand_list(
        (3, 3), length=ts
    ),  # [`ts` random ndarrays of shape (3,3)],
    "process_noise": rand_list(
        (3, 3), length=ts
    ),  # [`ts` random ndarrays of shape (3,3)],
    "measurement_noise": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "measurement_function": rand_list(
        (1, 3), length=ts
    ),  # [`ts` random ndarrays of shape (1,3)],
}


def init_kf_filterpy(measurements, adapter, n=10, y=None):
    """Adjust params and measurements.

    Given measurements and adapter, adjust params and measurements to
    `FilterPy` usable form.
    """
    y_dim = 1 if y is None else y.shape[-1]

    G = (
        np.eye(adapter.state_dim, y_dim)
        if adapter.control_transition is None
        else np.atleast_2d(adapter.control_transition)
    )

    matrices = {
        "Fs": [adapter.F_] * n if adapter.F_.ndim == 2 else list(adapter.F_),
        "Qs": [adapter.Q_] * n if adapter.Q_.ndim == 2 else list(adapter.Q_),
        "Rs": [adapter.R_] * n if adapter.R_.ndim == 2 else list(adapter.R_),
        "Hs": [adapter.H_] * n if adapter.H_.ndim == 2 else list(adapter.H_),
        "Bs": [G] * n if G.ndim == 2 else list(G),
        "us": None if y is None else [y] * n if y.ndim == 1 else list(y),
        "x": adapter.X0_,
        "P": adapter.P0_,
    }
    data = [None if any(np.isnan(d)) else d.copy() for d in measurements]

    return matrices, data


@pytest.mark.skipif(
    not _check_soft_dependencies(["filterpy", "numpy<1.24.0"], severity="none"),
    reason=(
        "skip test if required soft dependency filterpy not available "
        "or numpy version is greater than 1.24.0"
    ),
)
@pytest.mark.parametrize(
    "params, measurements, y",
    [  # test case 1 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_dynamic
        (dict(params_3_3_dynamic, state_dim=3), create_data((ts, 3)), None),
        # control_transition (aka G or B) is set with random ndarray of shape (3, 3).
        # y is set with random ndarray of shape (10, 3).
        # H and X0 will be estimated using em algorithm.
        (
            dict(
                params_3_3_dynamic,
                state_dim=3,
                control_transition=create_data((3, 3)),
                estimate_matrices=None,
            ),
            create_data((ts, 3)),
            create_data((ts, 3)),
        ),
        # test case 2 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_static
        # control_transition (aka G or B) is set with a list of 10 random
        # ndarrays, each of shape (3, 2).
        # y is set with random ndarray of shape (2,).
        (
            dict(params_3_3_static, state_dim=3, control_transition=rand_list((3, 2))),
            create_data((ts, 3)),
            create_data(2),
        ),
        # all matrix parameters will be estimated using em algorithm.
        # control_transition (aka G or B) is set with random ndarray of shape (3, 1).
        # y is set with random ndarray of shape (1,).
        (
            dict(
                params_3_3_static,
                state_dim=3,
                estimate_matrices=None,
                control_transition=create_data((3, 1)),
            ),
            create_data((ts, 3)),
            create_data(1),
        ),
        # test case 3 -
        #   state_dim = 2, measurement_dim = 3, params - params_2_3_,
        # y is set with random ndarray of shape (10, 4).
        (dict(params_2_3_, state_dim=2), create_data((ts, 3)), create_data((ts, 4))),
        # R will be estimated using em algorithm.
        (
            dict(params_2_3_, state_dim=2, estimate_matrices=None),
            create_data((ts, 3)),
            None,
        ),
        # test case 4 -
        #   state_dim = 3, measurement_dim = 1, params are None
        # control_transition (aka G or B) - is set with random ndarray of shape (3, 2).
        # should raise a warning and control_transition will be
        # ignored during calculation.
        (
            dict(state_dim=3, control_transition=create_data((3, 2))),
            create_data((ts, 1), missing_values=True),
            None,
        ),
        # F and Q will be estimated using em algorithm.
        (
            dict(state_dim=3, estimate_matrices=None),
            create_data((ts, 1), missing_values=True),
            None,
        ),
        # test case 5 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_arrays,
        # control_transition is set with random ndarray of shape (10, 1).
        # y is set with random ndarray of shape (10, 1).
        (
            dict(
                params_1_1_arrays,
                state_dim=1,
                control_transition=create_data((ts, 1, 1)),
            ),
            create_data((ts, 1)),
            create_data((ts, 1)),
        ),
        # P0 will be estimated using em algorithm.
        # control_transition is set with a list of 10 random ndarrays of shape (1,).
        # y is set with random ndarray of shape (10, 1).
        (
            dict(
                params_1_1_arrays,
                state_dim=1,
                estimate_matrices=None,
                control_transition=rand_list((1, 1)),
            ),
            create_data((ts, 1)),
            create_data((ts, 1)),
        ),
        # test case 6 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_lists
        # y is set with random ndarray of shape (2,).
        (
            dict(params_1_1_lists, state_dim=1),
            create_data((ts, 1), missing_values=True),
            create_data(2),
        ),
        # test case 7 -
        #   state_dim = 3, measurement_dim = 1, params - params_3_1_lists
        # control_transition is set with random ndarray of shape (3, 3).
        # y is set with random ndarray of shape (10, 1).
        (
            dict(params_3_1_lists, state_dim=3, control_transition=create_data((3, 3))),
            create_data((ts, 1), missing_values=True),
            create_data((ts, 3)),
        ),
    ],
)
def test_transform_and_smooth_fp(params, measurements, y):
    """Test KalmanFilterTransformer `fit` and `transform`.

    Creating two instances of KalmanFilterTransformer, one instance
    with parameter `denoising` set to False, and the other's set to True.
    Compare result with `FilterPy`'s `batch_filter` and `rts_smoother`.
    """
    from filterpy.kalman.kalman_filter import batch_filter, rts_smoother

    # initiate KalmanFilterTransformer with denoising=False
    # fit and transform
    adapter_transformer = KalmanFilterTransformer(**params)
    adapter_transformer = adapter_transformer.fit(measurements, y=y)
    xt_transformer_adapter = adapter_transformer.transform(measurements, y=y)

    # initiating KalmanFilterTransformer with denoising=True,
    # fit and transform
    adapter_smoother = KalmanFilterTransformer(denoising=True, **params)
    adapter_smoother = adapter_smoother.fit(measurements, y=y)
    xt_smoother_adapter = adapter_smoother.transform(measurements, y=y)

    # get data in a form compatible to FilterPy. call batch_filter
    matrices, fp_measurements = init_kf_filterpy(
        measurements=measurements, adapter=adapter_transformer, y=y
    )
    (means, covs, _, _) = batch_filter(**matrices, zs=fp_measurements)

    # test transformer
    assert_array_almost_equal(xt_transformer_adapter, means)

    # test smoother
    xt_smoother_filterpy = rts_smoother(
        Xs=means, Ps=covs, Fs=matrices["Fs"], Qs=matrices["Qs"]
    )[0]
    assert_array_almost_equal(xt_smoother_adapter, xt_smoother_filterpy)
