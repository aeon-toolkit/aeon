"""Soft elastic distance functions."""

__all__ = [
    "soft_dtw_alignment_matrix",
    "soft_dtw_alignment_path",
    "soft_dtw_cost_matrix",
    "soft_dtw_distance",
    "soft_dtw_pairwise_distance",
    "soft_msm_alignment_path",
    "soft_msm_distance",
    "soft_msm_cost_matrix",
    "soft_msm_pairwise_distance",
    "soft_bag_distance",
    "soft_bag_alignment_path",
    "soft_bag_cost_matrix",
    "soft_bag_pairwise_distance",
    "_soft_bag_alignment_matrix_univariate",
    "soft_bag_alignment_matrix",
    "soft_msm_alignment_matrix",
    "soft_dtw_grad_x",
    "soft_dtw_divergence_distance",
    "soft_dtw_divergence_pairwise_distance",
    "soft_dtw_divergence_grad_x",
    "soft_msm_divergence_distance",
    "soft_msm_divergence_pairwise_distance",
    "soft_msm_divergence_grad_x",
    "soft_msm_grad_x",
]
from aeon.distances.elastic.soft._soft_bag import (
    _soft_bag_alignment_matrix_univariate,
    soft_bag_alignment_matrix,
    soft_bag_alignment_path,
    soft_bag_cost_matrix,
    soft_bag_distance,
    soft_bag_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_dtw import (
    soft_dtw_alignment_matrix,
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_grad_x,
    soft_dtw_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_dtw_divergence import (
    soft_dtw_divergence_distance,
    soft_dtw_divergence_grad_x,
    soft_dtw_divergence_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_msm import (
    soft_msm_alignment_matrix,
    soft_msm_alignment_path,
    soft_msm_cost_matrix,
    soft_msm_distance,
    soft_msm_grad_x,
    soft_msm_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_msm_divergence import (
    soft_msm_divergence_distance,
    soft_msm_divergence_grad_x,
    soft_msm_divergence_pairwise_distance,
)
