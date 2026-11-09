"""Data generators."""

__all__ = [
    # collections
    "make_example_3d_numpy",
    "make_example_2d_numpy_collection",
    "make_example_3d_numpy_list",
    "make_example_2d_numpy_list",
    "make_example_dataframe_list",
    "make_example_2d_dataframe_collection",
    "make_example_multi_index_dataframe",
    # series
    "make_example_1d_numpy",
    "make_example_2d_numpy_series",
    "make_example_pandas_series",
    "make_example_dataframe_series",
    "make_example_mean_shift_series",
    "make_example_variance_shift_series",
    "make_example_multivariate_series",
    "make_mean_shift_series",
    "make_variance_shift_series",
    "make_multivariate_series",
    # labels
    "make_anomaly_detection_labels",
]


from aeon.testing.data_generation._collection import (
    make_example_2d_dataframe_collection,
    make_example_2d_numpy_collection,
    make_example_2d_numpy_list,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_list,
    make_example_multi_index_dataframe,
)
from aeon.testing.data_generation._labels import make_anomaly_detection_labels
from aeon.testing.data_generation._series import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_dataframe_series,
    make_example_mean_shift_series,
    make_example_multivariate_series,
    make_example_pandas_series,
    make_example_variance_shift_series,
    make_mean_shift_series,
    make_multivariate_series,
    make_variance_shift_series,
)
