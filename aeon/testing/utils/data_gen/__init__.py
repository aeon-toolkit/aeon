"""Data generators."""

__all__ = [
    "make_example_2d_numpy",
    "make_example_3d_numpy",
    "make_example_unequal_length",
    "make_example_nested_dataframe",
    "make_example_long_table",
    "make_example_multi_index_dataframe",
    "make_series",
    "make_forecasting_problem",
    "_make_index",
    "piecewise_normal_multivariate",
    "piecewise_normal",
    "piecewise_multinomial",
    "piecewise_poisson",
    "labels_with_repeats",
    "label_piecewise_normal",
    "_make_collection",
    "_make_collection_X",
    "_make_classification_y",
    "_make_hierarchical",
    "_bottom_hier_datagen",
    "_make_collection",
    "_make_nested_from_array",
    "_make_regression_y",
    "_make_fh",
    "_assert_correct_columns",
    "_assert_correct_pred_time_index",
    "make_annotation_problem",
    "_convert_tsf_to_hierarchical",
    "_get_n_columns",
    "get_examples",
    "make_example_2d_unequal_length",
]


from aeon.testing.utils.data_gen._collection import (
    _make_classification_y,
    _make_collection,
    _make_collection_X,
    _make_nested_from_array,
    _make_regression_y,
    make_example_2d_numpy,
    make_example_2d_unequal_length,
    make_example_3d_numpy,
    make_example_long_table,
    make_example_multi_index_dataframe,
    make_example_nested_dataframe,
    make_example_unequal_length,
)
from aeon.testing.utils.data_gen._data_generators import _convert_tsf_to_hierarchical
from aeon.testing.utils.data_gen._series import (
    _make_index,
    make_forecasting_problem,
    make_series,
)
from aeon.testing.utils.data_gen._test_examples import get_examples
from aeon.testing.utils.data_gen.annotation import make_annotation_problem
from aeon.testing.utils.data_gen.forecasting import (
    _assert_correct_columns,
    _assert_correct_pred_time_index,
    _get_n_columns,
    _make_fh,
)
from aeon.testing.utils.data_gen.hierarchical import (
    _bottom_hier_datagen,
    _make_hierarchical,
)
from aeon.testing.utils.data_gen.segmentation import (
    label_piecewise_normal,
    labels_with_repeats,
    piecewise_multinomial,
    piecewise_normal,
    piecewise_normal_multivariate,
    piecewise_poisson,
)
