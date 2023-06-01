# -*- coding: utf-8 -*-
"""Functions to load and write datasets."""

__all__ = [
    # Load functions
    "load_from_tsfile",
    "load_from_arff",
    "load_from_long_to_dataframe",
    "load_tsf_to_dataframe",
    "load_from_tsfile_to_dataframe",
    "load_from_ucr_tsv_to_dataframe",
    # Write functions
    "write_to_tsfile",
    "write_results_to_uea_format",
    # Data generators
    "make_example_long_table",
    "make_example_multi_index_dataframe",
    # Single problem loaders
    "load_TSC_dataset",
    "load_airline",
    "load_arrow_head",
    "load_gunpoint",
    "load_basic_motions",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_japanese_vowels",
    "load_plaid",
    "load_longley",
    "load_lynx",
    "load_shampoo_sales",
    "load_unit_test",
    "load_uschange",
    "load_PBS_dataset",
    "load_japanese_vowels",
    "load_gun_point_segmentation",
    "load_electric_devices_segmentation",
    "load_acsf1",
    "load_macroeconomic",
    "load_unit_test_tsf",
    "load_solar",
    "load_cardano_sentiment",
    "load_covid_3month",
]

from aeon.datasets._data_dataframe_loaders import (
    load_from_long_to_dataframe,
    load_from_ucr_tsv_to_dataframe,
    load_tsf_to_dataframe,
)
from aeon.datasets._data_generators import (
    make_example_long_table,
    make_example_multi_index_dataframe,
)
from aeon.datasets._data_loaders import load_from_arff, load_from_tsfile
from aeon.datasets._data_writers import write_results_to_uea_format, write_to_tsfile
from aeon.datasets._single_problem_loaders import (
    load_acsf1,
    load_airline,
    load_arrow_head,
    load_basic_motions,
    load_cardano_sentiment,
    load_covid_3month,
    load_electric_devices_segmentation,
    load_gun_point_segmentation,
    load_gunpoint,
    load_italy_power_demand,
    load_japanese_vowels,
    load_longley,
    load_lynx,
    load_macroeconomic,
    load_osuleaf,
    load_PBS_dataset,
    load_plaid,
    load_shampoo_sales,
    load_solar,
    load_TSC_dataset,
    load_unit_test,
    load_unit_test_tsf,
    load_uschange,
)
