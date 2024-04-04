import numpy as np
import pytest

from aeon.testing.mock_estimators._mock_anomaly_detectors import MockAnomalyDetector


def test_check_y():
    """Test the anomaly detection check_y method."""
    ad = MockAnomalyDetector()

    # Test for np.ndarray
    np_int = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
    np_bool = np.array(
        [False, False, False, True, True, False, False, False, False, True, True]
    )
    np_wrong_int = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 2])
    np_wrong_int2 = np.array([-1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1])
    np_2d_int = np.array(
        [
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        ]
    )
    np_float = np.array([0, 0, 0, 1.5, 1.5, 0, 0, 0, 0, 2.5, 2.5])

    new_np_int = ad._check_y(np_int, False)
    assert isinstance(new_np_int, np.ndarray)
    assert (new_np_int == np_bool).all()

    new_np_bool = ad._check_y(np_bool, False)
    assert isinstance(new_np_bool, np.ndarray)
    assert (new_np_bool == np_bool).all()

    with pytest.raises(ValueError, match="y input type must be an integer array"):
        ad._check_y(np_wrong_int, False)
    with pytest.raises(ValueError, match="y input type must be an integer array"):
        ad._check_y(np_wrong_int2, False)
    with pytest.raises(ValueError, match="y input as np.ndarray should be 1D"):
        ad._check_y(np_2d_int, False)
    with pytest.raises(ValueError, match="y input type must be an integer array"):
        ad._check_y(np_float, False)

    # Test for pd.Series
