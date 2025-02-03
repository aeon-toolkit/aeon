"""Test functions in yield estimator checks."""

import numpy as np

from aeon.testing.estimator_checking._yield_estimator_checks import _equal_outputs


def test_equal_outputs():
    """Test compare outputs for estimators.

    Valid data structures are:
    1. float: returns a single value (e.g. forecasting)
    2. boolean: returns true/false
    2. numpy array: stores an equal length collection or series (default)
    3. dict: a histogram of counts, usually of discretised sub-series (e.g. SFA)
    4. pd.DataFrame: series stored in dataframe (e.g. Dobin)
    5. list: stores possibly unequal length series in a format 2-4
    6. tuple: stores two or more series/collections in a format 2-4 (e.g. imbalance
    transformers)

    """
    v1 = 10.00
    v2 = 15.00
    assert _equal_outputs(v1, v1)
    assert not _equal_outputs(v1, v2)
    b1 = True
    b2 = False
    assert _equal_outputs(b1, b1)
    assert not _equal_outputs(b1, b2)
    b3 = np.array([True, True, False])
    b4 = np.array([True, False, False])
    assert _equal_outputs(b3, b3)
    assert not _equal_outputs(b3, b4)
    a1 = np.random.random((5, 1, 10))
    a2 = np.random.random((5, 10))
    assert _equal_outputs(a1, a1)
    assert not _equal_outputs(a1, a2)
    d1 = {"a": a1, "b": a2}
    d2 = {"a": a1, "b": a1}
    assert _equal_outputs(d1, d1)
    assert not _equal_outputs(d1, d2)
    t1 = (a1, a2)
    t2 = (a1, a1)
    assert _equal_outputs(t1, t1)
    assert not _equal_outputs(t1, t2)

    a1 = np.random.random((5, 10))
    a2 = np.random.random((5, 10))
    o1 = np.array([a2, a1], dtype=object)
    o2 = np.array([a1, a2], dtype=object)
    assert _equal_outputs(o1, o1)
    assert not _equal_outputs(o1, o2)
