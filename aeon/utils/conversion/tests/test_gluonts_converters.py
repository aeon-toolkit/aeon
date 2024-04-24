"""Tests for function in convert_from_multiindex_to_listdataset.py."""

from aeon.testing.utils.data_gen import make_example_multi_index_date_index
from aeon.utils.conversion.gluonts_converters import (
    convert_from_multiindex_to_listdataset,
)


def test_convert_from_multiindex_to_listdataset():
    """Test the onvert_from_multiindex_to_listdataset."""
    x = make_example_multi_index_date_index()
    y = convert_from_multiindex_to_listdataset(x)
    assert isinstance(y, list)
    assert len(y) == 50
