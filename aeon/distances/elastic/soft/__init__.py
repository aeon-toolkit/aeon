"""Soft elastic distance functions."""

__all__ = [
    "soft_dtw_alignment_matrix",
    "soft_dtw_alignment_path",
    "soft_dtw_cost_matrix",
    "soft_dtw_distance",
    "soft_dtw_pairwise_distance",
    "soft_dtw_grad_x",
    "soft_msm_alignment_matrix",
    "soft_msm_alignment_path",
    "soft_msm_cost_matrix",
    "soft_msm_distance",
    "soft_msm_pairwise_distance",
    "soft_msm_grad_x",
]

from aeon.distances.elastic.soft._soft_dtw import (
    soft_dtw_alignment_matrix,
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_grad_x,
    soft_dtw_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_msm import (
    soft_msm_alignment_matrix,
    soft_msm_alignment_path,
    soft_msm_cost_matrix,
    soft_msm_distance,
    soft_msm_grad_x,
    soft_msm_pairwise_distance,
)
