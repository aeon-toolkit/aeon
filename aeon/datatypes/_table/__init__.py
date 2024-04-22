"""Module exports: Series type checkers, converters and mtype inference."""

from aeon.datatypes._table._check import check_dict as check_dict_Table
from aeon.datatypes._table._convert import convert_dict as convert_dict_Table
from aeon.datatypes._table._examples import example_dict as example_dict_Table
from aeon.datatypes._table._examples import (
    example_dict_lossy as example_dict_lossy_Table,
)
from aeon.datatypes._table._examples import (
    example_dict_metadata as example_dict_metadata_Table,
)
from aeon.datatypes._table._registry import TYPE_LIST_TABLE, TYPE_REGISTER_TABLE

__all__ = [
    "check_dict_Table",
    "convert_dict_Table",
    "TYPE_LIST_TABLE",
    "TYPE_REGISTER_TABLE",
    "example_dict_Table",
    "example_dict_lossy_Table",
    "example_dict_metadata_Table",
]
