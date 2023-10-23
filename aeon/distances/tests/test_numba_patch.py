import numpy as np
import pytest

from aeon.distances._distance import DISTANCES


@pytest.mark.parametrize("dist", DISTANCES)
def test_numba_cache(dist):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    dist["distance"](x, y)
