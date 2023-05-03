# -*- coding: utf-8 -*-
"""Channel selection test code."""
import numpy as np

from aeon.transformations.panel.channel_selection import ElbowClassPairwise
from aeon.utils._testing.collection import make_3d_test_data


def test_channel_selection():
    """Test channel selection on random nested data frame."""
    X, y = make_3d_test_data(n_samples=10, n_channels=4, series_length=20)
    y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    ecp = ElbowClassPairwise()

    ecp.fit(X, y)
    Xt = ecp.transform(X, y)

    # test shape pf transformed data should be (n_samples, n_channels_selected,
    # series_length)
    assert Xt.shape == (X.shape[0], len(ecp.channels_selected_idx), X.shape[2])
