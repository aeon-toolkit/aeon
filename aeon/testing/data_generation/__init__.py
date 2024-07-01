"""Data generators."""

__all__ = [
    # collections
    "make_example_3d_numpy",
    "make_example_2d_numpy_collection",
    "make_example_3d_numpy_list",
    "make_example_2d_numpy_list",
    "make_example_dataframe_list",
    "make_example_2d_dataframe_collection",
    "make_example_nested_dataframe",
    "make_example_multi_index_dataframe",
    # series
    "make_example_1d_numpy",
    "make_example_2d_numpy_series",
    "make_example_pandas_series",
    "make_example_dataframe_series",
    # other
    "piecewise_normal_multivariate",
    "piecewise_normal",
    "piecewise_multinomial",
    "piecewise_poisson",
    "labels_with_repeats",
    "label_piecewise_normal",
    "_make_hierarchical",
    "_bottom_hier_datagen",
]


from aeon.testing.data_generation._collection import (
    make_example_2d_dataframe_collection,
    make_example_2d_numpy_collection,
    make_example_2d_numpy_list,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_list,
    make_example_multi_index_dataframe,
    make_example_nested_dataframe,
)
from aeon.testing.data_generation._series import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_dataframe_series,
    make_example_pandas_series,
)
from aeon.testing.data_generation.hierarchical import (
    _bottom_hier_datagen,
    _make_hierarchical,
)
from aeon.testing.data_generation.segmentation import (
    label_piecewise_normal,
    labels_with_repeats,
    piecewise_multinomial,
    piecewise_normal,
    piecewise_normal_multivariate,
    piecewise_poisson,
)
