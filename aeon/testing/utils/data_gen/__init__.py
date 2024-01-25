"""Data generators."""

__all__ = [
    "make_2d_test_data",
    "make_3d_test_data",
    "make_unequal_length_test_data",
    "make_series",
    "make_clustering_data",
    "piecewise_normal_multivariate",
    "piecewise_normal",
    "piecewise_multinomial",
    "piecewise_poisson",
    "labels_with_repeats",
    "label_piecewise_normal",
]


from aeon.testing.utils.data_gen._collection import (
    make_2d_test_data,
    make_3d_test_data,
    make_clustering_data,
    make_unequal_length_test_data,
)
from aeon.testing.utils.data_gen.series import make_series
from aeon.testing.utils.segmentation import (
    label_piecewise_normal,
    labels_with_repeats,
    piecewise_multinomial,
    piecewise_normal,
    piecewise_normal_multivariate,
    piecewise_poisson,
)
