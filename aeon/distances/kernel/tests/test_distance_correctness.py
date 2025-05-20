"""Test the distance calculations are correct.

Compare the distance calculations on the 1D and 2D (d,m) format input against the
results generated with the Octave implementation from
https://people.irisa.fr/Pierre-Francois.Marteau/REDK/KDTW/kdtw.m (adapted).
"""

from numpy.testing import assert_array_almost_equal_nulp

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.distances import kdtw_distance

distances = [
    "kdtw",
]

distance_parameters = {
    "kdtw": [1e-5, 1e-2, 1e-1],  # gamma
}
basic_motions_distances = {
    "kdtw": [1.0 - 2.7435e-123, 1.0 - 4.9180e-72, 1.0 - 6.5392e-53],
}
unit_test_distances = {
    "kdtw": [1.0 - 6.6667e-21, 1.0 - 6.6667e-21, 1.0 - 6.6667e-21],
}


def test_multivariate_correctness():
    """Test distance correctness on BasicMotions: multivariate, equal length."""
    trainX, _ = load_basic_motions(return_type="numpy3D")
    case1 = trainX[0]
    case2 = trainX[1]

    for j in range(0, 3):
        d = kdtw_distance(
            case1,
            case2,
            gamma=distance_parameters["kdtw"][j],
            normalize_input=False,
            normalize_dist=False,
        )
        assert_array_almost_equal_nulp(d, basic_motions_distances["kdtw"][j])


def test_univariate_correctness():
    """Test correctness on UnitTest: univariate, equal length."""
    trainX, _ = load_unit_test(return_type="numpy3D")
    trainX2, _ = load_unit_test(return_type="numpy2D")
    # Test 2D and 3D instances from UnitTest
    cases1 = [trainX[0], trainX2[0]]
    cases2 = [trainX[2], trainX2[2]]

    for j in range(0, 3):
        d = kdtw_distance(
            cases1[0],
            cases2[0],
            gamma=distance_parameters["kdtw"][j],
            normalize_input=False,
            normalize_dist=False,
        )
        d2 = kdtw_distance(
            cases1[1],
            cases2[1],
            gamma=distance_parameters["kdtw"][j],
            normalize_input=False,
            normalize_dist=False,
        )
        assert_array_almost_equal_nulp(d, unit_test_distances["kdtw"][j])
        assert d == d2
