"""Test functions for input validation."""

from aeon.testing.utils.data_gen import get_examples

# is_collection,; is_single_series,; is_valid_input,
from aeon.utils.validation._input import is_hierarchical


def test_is_hierarchical():
    res = get_examples("pd_multiindex_hier")
    for r in res:
        assert is_hierarchical(r)
