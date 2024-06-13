"""Data generators."""

__all__ = [
    # collections
    "make_example_3d_numpy",
    "make_example_2d_numpy",
    "make_example_3d_numpy_list",
    "make_example_2d_numpy_list",
    "make_example_dataframe_list",
    "make_example_2d_dataframe",
    "make_example_nested_dataframe",
    "make_example_multi_index_dataframe",
    # other
    "make_series",
    "make_forecasting_problem",
    "_make_index",
    "piecewise_normal_multivariate",
    "piecewise_normal",
    "piecewise_multinomial",
    "piecewise_poisson",
    "labels_with_repeats",
    "label_piecewise_normal",
    "_make_hierarchical",
    "_bottom_hier_datagen",
    "_make_fh",
    "_assert_correct_columns",
    "_assert_correct_pred_time_index",
    "_get_n_columns",
    "get_examples",
]


from aeon.testing.data_generation._collection import (
    make_example_2d_dataframe,
    make_example_2d_numpy,
    make_example_2d_numpy_list,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_list,
    make_example_multi_index_dataframe,
    make_example_nested_dataframe,
)
from aeon.testing.data_generation._series import (
    _make_index,
    make_forecasting_problem,
    make_series,
)
from aeon.testing.data_generation._test_examples import get_examples
from aeon.testing.data_generation.forecasting import (
    _assert_correct_columns,
    _assert_correct_pred_time_index,
    _get_n_columns,
    _make_fh,
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
