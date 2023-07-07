# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Functions to load and write datasets."""

__all__ = [
    # Load/download functions
    "load_from_tsfile",
    "load_from_tsf_file",
    "load_from_arff_file",
    "load_from_tsv_file",
    "load_classification",
    "load_forecasting",
    "load_regression",
    "download_all_regression",
    # Write functions
    "write_to_tsfile",
    "write_results_to_uea_format",
    # Data generators
    "make_example_3d_numpy",
    "make_example_2d_numpy",
    "make_example_long_table",
    "make_example_multi_index_dataframe",
    # Single problem loaders
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
    # legacy load functions
    "load_from_long_to_dataframe",
    "load_tsf_to_dataframe",
    "load_from_tsfile_to_dataframe",
]

from aeon.datasets._data_generators import (
    make_example_2d_numpy,
    make_example_3d_numpy,
    make_example_long_table,
    make_example_multi_index_dataframe,
)
from aeon.datasets._data_loaders import (
    download_all_regression,
    load_classification,
    load_forecasting,
    load_from_arff_file,
    load_from_tsf_file,
    load_from_tsfile,
    load_from_tsv_file,
    load_regression,
)
from aeon.datasets._data_writers import write_results_to_uea_format, write_to_tsfile
from aeon.datasets._dataframe_loaders import (
    load_from_long_to_dataframe,
    load_from_tsfile_to_dataframe,
    load_tsf_to_dataframe,
)
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
    load_unit_test,
    load_unit_test_tsf,
    load_uschange,
)
