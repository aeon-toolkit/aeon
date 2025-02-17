"""Soft distances."""

__all__ = [
    "soft_dtw_alignment_path",
    "soft_dtw_cost_matrix",
    "soft_dtw_distance",
    "soft_dtw_pairwise_distance",
    "soft_dtw_gradient",
    "soft_twe_alignment_path",
    "soft_twe_cost_matrix",
    "soft_twe_distance",
    "soft_twe_pairwise_distance",
    "soft_twe_gradient",
    "soft_msm_alignment_path",
    "soft_msm_cost_matrix",
    "soft_msm_distance",
    "soft_msm_pairwise_distance",
    "soft_msm_gradient",
    "soft_adtw_distance",
    "soft_adtw_cost_matrix",
    "soft_adtw_alignment_path",
    "soft_adtw_gradient",
    "soft_adtw_pairwise_distance",
    "soft_shape_dtw_alignment_path",
    "soft_shape_dtw_cost_matrix",
    "soft_shape_dtw_distance",
    "soft_shape_dtw_pairwise_distance",
    "soft_shape_dtw_gradient",
    "soft_wdtw_alignment_path",
    "soft_wdtw_cost_matrix",
    "soft_wdtw_distance",
    "soft_wdtw_pairwise_distance",
    "soft_wdtw_gradient",
    "soft_erp_distance",
    "soft_erp_cost_matrix",
    "soft_erp_alignment_path",
    "soft_erp_gradient",
    "soft_erp_pairwise_distance",
]

from aeon.distances.elastic.soft._soft_adtw import (
    soft_adtw_alignment_path,
    soft_adtw_cost_matrix,
    soft_adtw_distance,
    soft_adtw_gradient,
    soft_adtw_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_dtw import (
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_gradient,
    soft_dtw_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_erp import (
    soft_erp_alignment_path,
    soft_erp_cost_matrix,
    soft_erp_distance,
    soft_erp_gradient,
    soft_erp_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_msm import (
    soft_msm_alignment_path,
    soft_msm_cost_matrix,
    soft_msm_distance,
    soft_msm_gradient,
    soft_msm_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_shape_dtw import (
    soft_shape_dtw_alignment_path,
    soft_shape_dtw_cost_matrix,
    soft_shape_dtw_distance,
    soft_shape_dtw_gradient,
    soft_shape_dtw_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_twe import (
    soft_twe_alignment_path,
    soft_twe_cost_matrix,
    soft_twe_distance,
    soft_twe_gradient,
    soft_twe_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_wdtw import (
    soft_wdtw_alignment_path,
    soft_wdtw_cost_matrix,
    soft_wdtw_distance,
    soft_wdtw_gradient,
    soft_wdtw_pairwise_distance,
)
