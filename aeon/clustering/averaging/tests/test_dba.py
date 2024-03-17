"""Tests for DBA."""

import numpy as np
import pytest

from aeon.clustering.averaging import elastic_barycenter_average
from aeon.testing.utils.data_gen import make_example_3d_numpy

expected_dba = np.array(
    [
        [
            2.14365783,
            1.79294991,
            1.40583643,
            1.41551047,
            1.35200733,
            1.89139937,
            1.86542327,
            1.32466944,
            1.67397517,
            1.8709378,
        ],
        [
            1.39792187,
            1.48779949,
            1.26747876,
            1.77152767,
            1.89443306,
            1.33528121,
            1.8849385,
            1.61457405,
            1.6470203,
            1.58740109,
        ],
        [
            1.20289266,
            0.99392693,
            0.91151178,
            1.36251883,
            1.63591345,
            1.76292241,
            1.2924362,
            1.82067719,
            1.36224209,
            2.09249394,
        ],
        [
            1.50362661,
            1.33955338,
            2.21909116,
            1.84855474,
            1.31602877,
            1.53317511,
            1.84644465,
            1.42878873,
            1.78336225,
            1.59598453,
        ],
        [
            1.76544264,
            1.41121261,
            1.63573482,
            1.02127341,
            1.41004018,
            1.70329872,
            1.58462695,
            1.59081868,
            1.44386256,
            1.07904533,
        ],
        [
            1.30443531,
            1.58311813,
            1.70761705,
            2.20491277,
            1.26669821,
            1.70208611,
            1.91908395,
            1.60117971,
            1.90385381,
            1.64541667,
        ],
        [
            1.20010098,
            2.06998292,
            1.75802436,
            1.55995949,
            1.8326049,
            1.05988814,
            2.31728009,
            1.47100396,
            2.19823014,
            1.19986879,
        ],
        [
            1.55797104,
            1.40613571,
            1.85809765,
            1.41017291,
            1.9308945,
            1.37990633,
            1.77000225,
            1.35118687,
            1.25940145,
            1.85597143,
        ],
        [
            1.85856132,
            2.09754471,
            1.47383472,
            1.51400218,
            1.43779312,
            1.69484847,
            1.95836839,
            1.62152772,
            1.26513914,
            2.08603391,
        ],
        [
            1.2279546,
            1.53877513,
            1.32957377,
            1.71607611,
            1.46982699,
            1.95766663,
            1.4674824,
            1.85898204,
            1.81942838,
            1.6760823,
        ],
    ]
)


def test_dba():
    """Test dba functionality."""
    X_train = make_example_3d_numpy(10, 10, 10, random_state=1, return_y=False)

    average_ts = elastic_barycenter_average(X_train)

    assert isinstance(average_ts, np.ndarray)
    assert average_ts.shape == X_train[0].shape
    assert np.allclose(average_ts, expected_dba)


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

    assert isinstance(average_ts, np.ndarray)
    assert average_ts.shape == X_train[0].shape
