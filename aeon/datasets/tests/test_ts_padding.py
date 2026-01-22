"""Tests for padding missing dimensions in TS file loading."""

from io import StringIO

import numpy as np

from aeon.datasets._data_loaders import _load_data


def test_missing_dimension_is_padded():
    """Test that missing dimensions are padded with NaNs instead of raising."""
    ts = """@problemName test
@timestamps false
@univariate false
@dimensions 2
@data
1,2,3
"""

    file = StringIO(ts)
    meta = {"dimensions": 2}

    X, _, _ = _load_data(file, meta)

    assert X.shape == (1, 2, 3)
    assert np.isnan(X[0, 1]).all()
