import numpy as np

from aeon.transformations.collection.reduce import Tabularizer


def test_tabularizer():
    """Test Tabularizer."""
    tab = Tabularizer()
    arr = np.random.random(size=(10, 3, 100))
    res = tab.fit_transform(arr)
    assert res.shape == (10, 300)
    res = tab.fit_transform(arr)
    assert res.shape == (10, 300)
