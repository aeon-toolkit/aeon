"""Test the distance calculations are correct.

Compare the distance calculations on the 1D and 2D (d,m) format input against the
results generated with tsml, in distances.tests.TestDistances.
"""

from numpy.testing import assert_almost_equal

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.distances import kdtw_distance

distances = [
    "kdtw",
]

distance_parameters = {
    "kdtw": [0.0, 0.1, 1.0],  # gamma
}
unit_test_distances = {
    "kdtw": [0.0, 0.0, 0.0],
    "kdtw_norm": [0.0, 0.0, 0.0],
}
basic_motions_distances = {
    "kdtw": [0.0, 0.0, 0.0],
    "kdtw_norm": [0.0, 0.0, 0.0],
}


def test_multivariate_correctness():
    """Test distance correctness on BasicMotions: multivariate, equal length."""
    trainX, _ = load_basic_motions(return_type="numpy3D")
    case1 = trainX[0]
    case2 = trainX[1]

    for j in range(0, 3):
        d = kdtw_distance(
            case1, case2, gamma=distance_parameters["kdtw"][j], normalize=False
        )
        assert_almost_equal(d, basic_motions_distances["kdtw"][j], 4)
        d = kdtw_distance(
            case1, case2, gamma=distance_parameters["kdtw"][j], normalize=True
        )
        assert_almost_equal(d, basic_motions_distances["kdtw_norm"][j], 4)


def test_univariate_correctness():
    """Test correctness on UnitTest: univariate, equal length."""
    trainX, _ = load_unit_test(return_type="numpy3D")
    trainX2, _ = load_unit_test(return_type="numpy2D")
    # Test 2D and 3D instances from UnitTest
    cases1 = [trainX[0], trainX2[0]]
    cases2 = [trainX[2], trainX2[2]]

    for j in range(0, 3):
        d = kdtw_distance(
            cases1[0], cases2[0], gamma=distance_parameters["kdtw"][j], normalize=False
        )
        d2 = kdtw_distance(
            cases1[1], cases2[1], gamma=distance_parameters["kdtw"][j], normalize=False
        )
        assert_almost_equal(d, unit_test_distances["kdtw"][j], 4)
        assert d == d2
        d = kdtw_distance(
            cases1[0], cases2[0], gamma=distance_parameters["kdtw"][j], normalize=True
        )
        d2 = kdtw_distance(
            cases1[1], cases2[1], gamma=distance_parameters["kdtw"][j], normalize=True
        )
        assert_almost_equal(d, unit_test_distances["kdtw_norm"][j], 4)
        assert d == d2
