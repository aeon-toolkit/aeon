import numpy as np
import pandas as pd

from aeon.transformations.series import MatrixProfileTransformer


def test_matrix_profile():
    """Test on example in stumpy documentation."""
    series = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    series2 = pd.Series([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    mp = MatrixProfileTransformer(window_length=3)
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
