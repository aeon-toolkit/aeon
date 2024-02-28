import numpy as np
import pandas as pd
import pytest
from aeon.transformations.series import MatrixProfileSeriesTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies
import aeon.testing.utils._cicd_numba_caching

@pytest.mark.skipif(
    not _check_soft_dependencies("stumpy", severity="none"),
    reason="skip test if required soft dependency stumpy is not available",
)
def test_matrix_profile():
    """Test on example in stumpy documentation."""
    series = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
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

# Now, when you import your library, any usage of @jit will have cache=True enforced

# if __name__ == "__main__":
#
#     import numba.core.decorators
#     # Capture the original jit function
#     original_jit = numba.core.decorators._jit
#
#
#     def custom_njit(*args, **kwargs):
#         """Custom wrapper for numba's jit decorator to enforce caching."""
#         # Check if the first argument is a callable (i.e., a function), indicating that
#         # jit is used as a decorator without parentheses.
#         # kwargs.update({"cache": True})
#         target = kwargs['targetoptions']
#         if 'no_cpython_wrapper' not in target:
#             kwargs["cache"] = True
#         return original_jit(*args, **kwargs)
#
#
#     # Overwrite the jit function with the custom version
#     numba.core.decorators._jit = custom_njit
#
#     from aeon.distances import dtw_distance
#     from aeon.transformations.series import MatrixProfileSeriesTransformer
#
#     print("Starting")
#     dtw_distance(np.array([1, 2, 3]), np.array([1, 2, 3]))
#
#     series = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
#     series2 = pd.Series([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
#     mp = MatrixProfileSeriesTransformer(window_length=3)
#     res1 = mp.fit_transform(series)
#     res2 = mp.fit_transform(series2)
#     expected = np.array(
#         [
#             0.11633857113691416,
#             2.694073918063438,
#             3.0000926340485923,
#             2.694073918063438,
#             0.11633857113691416,
#         ]
#     )
#     np.testing.assert_allclose(res1, expected, rtol=1e-04, atol=1e-04)
#     np.testing.assert_allclose(res1, res2, rtol=1e-04, atol=1e-04)