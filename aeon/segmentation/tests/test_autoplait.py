"""Simple AutoPlait test."""

import pytest

from aeon.datasets import load_gun_point_segmentation
from aeon.segmentation import AutoPlaitSegmenter, _autoplait


def test_autoplait_sparse():
    """Test AutoPlait segmentation.

    Check if the predicted change points match.
    """

    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a FLUSS segmentation
    autoplait = AutoPlaitSegmenter()
    #found_cps = autoplait.fit_predict(ts)

    pass

def test_cut_point_search():
    X, regime1, regime2, d = [], [], [], []
    _autoplait._cut_point_search(X, regime1, regime2, d)
    pass

def test_regime_split():
    X = []
    _autoplait._regime_split(X)
    pass

def test_full_autoplait():
    X = []
    _autoplait._autoplait(X)
    pass