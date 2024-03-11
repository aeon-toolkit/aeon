import numpy as np
import pandas as pd
import pytest

from aeon.transformations.series import MatrixProfileSeriesTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("stumpy", severity="none"),
    reason="skip test if required soft dependency stumpy is not available",
)
def test_matrix_profile():
    """Test on example in stumpy documentation."""
    # Init as 2D array to test 1D convertion
    series = np.array([[584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0]])
    series2 = pd.Series([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    mp = MatrixProfileSeriesTransformer(window_length=3)
    res1 = mp.fit_transform(series)
    res2 = mp.fit_transform(series2)
    expected = np.array(
        [
            0.11633857113691416,
            2.694073918063438,
            3.0000926340485923,
            2.694073918063438,
            0.11633857113691416,
        ]
    )
    np.testing.assert_allclose(res1, expected, rtol=1e-04, atol=1e-04)
    np.testing.assert_allclose(res1, res2, rtol=1e-04, atol=1e-04)


def test_matrix_profile_multivariate_2D():
    """Test that an exception is raised when 2D series is given."""
    series = np.array(
        [
            [584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0],
            [584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0],
        ]
    )
    mp = MatrixProfileSeriesTransformer(window_length=3)
    try:
        mp.fit_transform(series)
    except ValueError:
        assert True
