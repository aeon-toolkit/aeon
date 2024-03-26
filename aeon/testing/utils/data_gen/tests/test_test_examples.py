"""Test examples."""

import pandas as pd

from aeon.testing.utils.data_gen._test_examples import (
    get_examples,
    get_hierarchical_examples,
    get_series_examples,
)
from aeon.utils.validation import is_collection


def test_get_examples():
    """Test get examples."""
    s1 = get_examples("pd.Series")
    assert isinstance(s1[0], pd.Series)
    s2 = get_series_examples()
    assert isinstance(s2, list)
    s1 = get_examples("pd_multiindex_hier")
    s2 = get_hierarchical_examples()
    assert s1 == s2
    s2 = get_examples("pd-multiindex")
    for s in s2:
        assert is_collection(s)
