""" Tests for MovingAverageTransformer """

__maintainer__ = ["Datadote"]

import numpy as np
from numpy.testing import assert_array_almost_equal

from aeon.transformations.series._moving_average import MovingAverageTransformer

TEST_DATA = [
    np.array([-3, -2, -1,  0,  1,  2,  3]),
    np.array([[-3, -2, -1,  0,  1,  2,  3]])
]
EXPECTED_RESULTS = [
    np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]]),
    np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]])
]

def test_moving_average():
    ma = MovingAverageTransformer(window_size=2)
    for i in range(len(TEST_DATA)):
        xt = ma.fit_transform(TEST_DATA[i])
        assert_array_almost_equal(xt, EXPECTED_RESULTS[i], decimal=2)