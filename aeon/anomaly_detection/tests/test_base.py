"""Test anomaly detection base class."""

import numpy as np
import pandas as pd
import pytest

from aeon.testing.mock_estimators import (
    MockAnomalyDetector,
    MockAnomalyDetectorRequiresY,
)

test_y = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])


def test_check_y():
    """Test the anomaly detection _check_y method."""
    ad = MockAnomalyDetector()
    ad_y = MockAnomalyDetectorRequiresY()
    correct_size = len(test_y)

    # Test for np.ndarray
    np_int = test_y
    np_wrong_int = np.array([0, 0, 0, 1, 1, 0, 0, 0, 2, 2])
    np_wrong_int2 = np.array([-1, -1, -1, 1, 1, -1, -1, -1, 1, 1])
    np_bool = np.array(
        [False, False, False, True, True, False, False, False, True, True]
    )
    np_2d_int = np.array(
        [
            [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        ]
    )
    np_float = np.array([0, 0, 0, 1.5, 1.5, 0, 0, 0, 2.5, 2.5])

    new_np_int = ad._check_y(np_int, correct_size)
    assert isinstance(new_np_int, np.ndarray)
    assert new_np_int.shape == (10,)
    assert issubclass(new_np_int.dtype.type, np.bool_)
    assert (new_np_int == np_bool).all()

    new_np_bool = ad._check_y(np_bool, correct_size)
    assert isinstance(new_np_bool, np.ndarray)
    assert new_np_bool.shape == (10,)
    assert issubclass(new_np_bool.dtype.type, np.bool_)
    assert (new_np_bool == np_bool).all()

    with pytest.raises(ValueError):
        ad._check_y(np_wrong_int, correct_size)
    with pytest.raises(ValueError):
        ad._check_y(np_wrong_int2, correct_size)
    with pytest.raises(TypeError):
        ad._check_y(np_2d_int, correct_size)
    with pytest.raises(ValueError):
        ad._check_y(np_float, correct_size)

    # Test for pd.Series
    pd_int = pd.Series(np_int)
    pd_wrong_int = pd.Series(np_wrong_int)
    pd_bool = pd.Series(np_bool)
    pd_float = pd.Series(np_float)

    new_pd_int = ad._check_y(pd_int, correct_size)
    assert isinstance(new_pd_int, np.ndarray)
    assert new_pd_int.shape == (10,)
    assert issubclass(new_pd_int.dtype.type, np.bool_)
    assert (new_pd_int == np_bool).all()

    new_pd_bool = ad._check_y(pd_bool, correct_size)
    assert isinstance(new_pd_bool, np.ndarray)
    assert new_pd_bool.shape == (10,)
    assert issubclass(new_pd_bool.dtype.type, np.bool_)
    assert (new_pd_bool == np_bool).all()

    with pytest.raises(ValueError):
        ad._check_y(pd_wrong_int, correct_size)
    with pytest.raises(ValueError):
        ad._check_y(pd_float, correct_size)

    # Test for pd.DataFrame
    pd_df_int = pd.DataFrame(np_int)
    pd_df_wrong_int = pd.DataFrame(np_wrong_int)
    pd_df_2d_int = pd.DataFrame(np_2d_int)
    pd_df_bool = pd.DataFrame(np_bool)
    pd_df_float = pd.DataFrame(np_float)

    new_pd_df_int = ad._check_y(pd_df_int, correct_length=correct_size)
    assert isinstance(new_pd_df_int, np.ndarray)
    assert new_pd_df_int.shape == (10,)
    assert issubclass(new_pd_df_int.dtype.type, np.bool_)
    assert (new_pd_df_int == np_bool).all()

    new_pd_df_bool = ad._check_y(pd_df_bool, correct_length=correct_size)
    assert isinstance(new_pd_df_bool, np.ndarray)
    assert new_pd_df_bool.shape == (10,)
    assert issubclass(new_pd_df_bool.dtype.type, np.bool_)
    assert (new_pd_df_bool == np_bool).all()

    with pytest.raises(ValueError):
        ad._check_y(pd_df_wrong_int, correct_size)
    with pytest.raises(TypeError, match="y input as pd.DataFrame should have"):
        ad._check_y(pd_df_2d_int, correct_size)
    with pytest.raises(ValueError):
        ad._check_y(pd_df_float, correct_size)

    # Test length check
    with pytest.raises(ValueError, match="Mismatch in number of labels"):
        ad._check_y(np_int, 0)

    # Test requires y
    with pytest.raises(ValueError, match="does not require a y input"):
        ad_y._check_y(np_float, correct_size)
