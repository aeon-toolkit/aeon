"""Soft elastic distance functions."""

__all__ = [
    "soft_dtw_alignment_matrix",
    "soft_dtw_alignment_path",
    "soft_dtw_cost_matrix",
    "soft_dtw_distance",
    "soft_dtw_pairwise_distance",
    "soft_dtw_grad_x",
]

from aeon.distances.elastic.soft._soft_dtw import (
    soft_dtw_alignment_matrix,
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_grad_x,
    soft_dtw_pairwise_distance,
)
