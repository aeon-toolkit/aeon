"""Tests for DOBIN (Distance based Outlier BasIs using Neighbors)."""

__maintainer__ = []

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from aeon.transformations.dobin import DOBIN


def test_fit_default():
    """Test with default parameters and high dimensional data."""
    X = np.array(
        [
            [14.374, 0.075, -0.689, -1.805, 0.370, -0.636],
            [15.184, -1.989, -0.707, 1.466, 0.267, -0.462],
            [14.164, 0.620, 0.365, 0.153, -0.543, 1.432],
            [16.595, -0.056, 0.769, 2.173, 1.208, -0.651],
            [15.330, -0.156, -0.112, 0.476, 1.160, -0.207],
            [14.180, -1.471, 0.881, -0.710, 0.700, -0.393],
            [15.487, -0.478, 0.398, 0.611, 1.587, -0.320],
            [15.738, 0.418, -0.612, -0.934, 0.558, -0.279],
            [15.576, 1.359, 0.341, -1.254, -1.277, 0.494],
            [14.695, -0.103, -1.129, 0.291, -0.573, -0.177],
            [0.302, 0.388, 1.433, -0.443, -1.225, -0.506],
            [0.078, -0.054, 1.980, 0.001, -0.473, 1.343],
            [-15.621, -1.377, -0.367, 0.074, -0.620, -0.215],
            [-17.215, -0.415, -1.044, -0.590, 0.042, -0.180],
            [-13.875, -0.394, 0.570, -0.569, -0.911, -0.100],
            [-15.045, -0.059, -0.135, -0.135, 0.158, 0.713],
            [-15.016, 1.100, 2.402, 1.178, -0.655, -0.074],
            [-14.056, 0.763, -0.039, -1.524, 1.767, -0.038],
            [-14.179, -0.165, 0.690, 0.594, 0.717, -0.682],
            [-14.406, -0.253, 0.028, 0.333, 0.910, -0.324],
            [-14.081, 0.697, -0.743, 1.063, 0.384, 0.060],
            [-14.218, 0.557, 0.189, -0.304, 1.682, -0.589],
        ]
    )

    # new coords
    coords_expected = np.array(
        [
            [0.70483267, -0.5816322, 0.11871428, 0.34552108, -0.09910254, -0.76377905],
            [0.75598487, -0.3995273, 0.77315224, 0.44127426, -0.57290326, -0.15798631],
            [1.39809346, -0.22739085, -0.00691087, 0.95976299, 0.02976105, -0.14004574],
            [1.28905629, -0.13633925, 0.68369824, 0.42504918, -0.71429055, -0.67108882],
            [1.03959645, -0.30184133, 0.54238311, 0.56120644, -0.31085749, -0.67220808],
            [1.01160787, -0.41870196, 0.57815941, 0.09295856, -0.07118286, -0.42409818],
            [1.12021801, -0.26005664, 0.70957693, 0.43559111, -0.29484726, -0.71200991],
            [0.88029154, -0.48277862, 0.17959014, 0.55378539, -0.16981412, -0.74845163],
            [
                1.23195092,
                -0.4504135,
                -0.4309906,
                0.62727426,
                -0.06502084,
                -0.37224069,
            ],
            [0.7829248, -0.45645, 0.12841989, 0.6655479, -0.45474935, -0.32971279],
            [1.08578146, -0.01800754, -0.24099512, 0.04146881, -0.3039681, -0.30324994],
            [1.48556002, 0.18778632, 0.05804308, 0.52291603, 0.21299198, -0.00437453],
            [0.43514895, 0.27618796, 0.20320291, 0.2480942, -0.17456449, -0.04190551],
            [0.33146677, 0.29603738, 0.10524435, 0.43712153, -0.05132798, -0.39590923],
            [0.71628228, 0.27462297, -0.09483719, 0.17156286, -0.06791329, -0.15334237],
            [0.75112256, 0.41796833, 0.11783, 0.63713402, 0.1198619, -0.27663729],
            [1.34017172, 0.67683558, -0.17428047, 0.15828576, -0.35301384, -0.32873135],
            [0.68278866, 0.35335046, 0.20063576, 0.43622768, 0.27183728, -0.99094384],
            [0.7683249, 0.46214507, 0.32043815, 0.14683553, -0.30481544, -0.58480058],
            [0.65894632, 0.42632203, 0.35897602, 0.34393226, -0.16654671, -0.57096877],
            [0.67726471, 0.47746418, 0.13025286, 0.72260868, -0.32161648, -0.52506194],
            [0.70015571, 0.43936681, 0.32210838, 0.29974983, -0.08728308, -0.96448606],
        ]
    )

    # rotation vector
    basis_expected = np.array(
        [
            [0.42165006, -0.86068265, 0.15102382, 0.14601341, -0.18165909, -0.06563891],
            [0.29467119, 0.16927356, -0.63128634, 0.33722485, -0.15896088, -0.58907024],
            [0.75280319, 0.23218556, -0.00239167, -0.59549543, 0.14261205, 0.06647174],
            [0.2451043, 0.36938044, 0.34480381, 0.3022277, -0.73490703, 0.23056968],
            [0.05236588, 0.16004401, 0.67794758, 0.11374533, 0.25620652, -0.65836998],
            [0.32534469, 0.1208541, 0.01269093, 0.63723237, 0.56183902, 0.39705903],
        ]
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = DOBIN()
    fitted_model = model.fit(X)
    coords_actual = fitted_model._coords
    basis_actual = fitted_model._basis

    assert np.allclose(coords_expected, coords_actual)
    assert np.allclose(basis_expected, basis_actual)


def test_fit_median_standardization():
    """Test with median standardisation and different parameters."""
    X = np.array(
        [
            [14.61, -0.11, 2.4],
            [14.94, 0.88, -0.04],
            [16.1, 0.4, 0.69],
            [15.76, -0.61, 0.03],
            [14.84, 0.34, -0.74],
            [-0.05, -1.13, 0.19],
            [0.14, 1.43, -1.8],
            [-14.44, 1.98, 1.47],
            [-15.69, -0.37, 0.15],
            [-15.71, -1.04, 2.17],
            [-14.64, 0.57, 0.48],
            [-14.23, -0.14, -0.71],
        ]
    )

    basis_expected = np.array(
        [
            [0.35798869, -0.84748114, -0.39194364],
            [0.25853965, 0.49331887, -0.83053822],
            [0.89721867, 0.19599032, 0.39571005],
        ]
    )

    coords_expected = np.array(
        [
            [1.95502872, -0.12345243, 0.7866799],
            [0.19274086, -0.11744806, -0.86460197],
            [0.69122866, -0.23973968, -0.2456952],
            [-0.09728797, -0.81074023, 0.29829762],
            [-0.51292537, -0.4873701, -0.70057837],
            [-0.28346375, -0.56367591, 0.96815996],
            [-1.30118311, 0.24590192, -1.72841774],
            [1.33847739, 1.50526326, -0.77327474],
            [-0.32468861, 0.22863203, 0.57668583],
            [1.17323565, 0.28483923, 1.82504575],
            [0.18467707, 0.68788554, -0.0423587],
            [-0.95797397, 0.13750341, 0.06841071],
        ]
    )

    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    model = DOBIN(k=3)
    fitted_model = model.fit(X)
    coords_actual = fitted_model._coords
    basis_actual = fitted_model._basis

    assert np.allclose(coords_expected, coords_actual)
    assert np.allclose(basis_expected, basis_actual)


def test_pca_reduction():
    """Test when n_obs is less than n_dim and PCA is applied initially."""
    X = np.array(
        [
            [1.4374e01, 6.5900e00, 1.1520e00, -1.2400e-01, -1.6000e-02, 9.1900e-01],
            [3.7000e-02, -1.6409e01, -6.1100e-01, -4.4300e-01, 9.4400e-01, 7.8200e-01],
            [-1.6700e-01, 9.7490e00, 3.0240e00, 2.2500e-01, 8.2100e-01, 7.5000e-02],
            [-1.3405e01, 1.4766e01, 7.8000e-01, -9.0000e-03, 5.9400e-01, -1.9890e00],
        ]
    )

    X_expected = np.array(
        [
            [0.2175, -0.805, 0.2853, 0.0],
            [0.9628, 0.4449, -0.1128, -0.0],
            [-0.5299, -0.1585, -0.5455, -0.0],
            [-0.6503, 0.5186, 0.3729, 0.0],
        ]
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = DOBIN(k=2, frac=0.5)
    fitted_model = model.fit(X)
    X_actual = fitted_model._X_pca

    assert np.allclose(abs(X_expected), abs(X_actual), rtol=0.001)


def test_zero_variance():
    """Test for column with zero variance."""
    X = np.array(
        [
            [-0.626, 0.659, 1.0],
            [0.184, -1.641, 1.0],
            [-0.836, 0.975, 1.0],
            [1.595, 1.477, 1.0],
        ]
    )

    coords_expected = np.array(
        [
            [0.4736931, 0.57202103, 0],
            [0.35209305, -0.22820652, 0],
            [0.45632521, 0.70405059, 0],
            [1.3830473, 0.2952629, 0],
        ]
    )

    basis_expected = np.array(
        [[0.8391551, -0.5438922, 0.0], [0.5438922, 0.8391551, 0.0], [0.0, 0.0, 1.0]]
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = DOBIN(k=1, frac=0.5)
    fitted_model = model.fit(X)
    coords_actual = fitted_model._coords
    basis_actual = fitted_model._basis

    assert np.allclose(coords_expected, coords_actual)
    assert np.allclose(basis_expected, basis_actual)
