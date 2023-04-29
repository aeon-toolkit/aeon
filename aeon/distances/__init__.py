# -*- coding: utf-8 -*-
"""Distance computation."""
__author__ = ["chrisholder", "TonyBagnall"]
__all__ = [
    "create_bounding_matrix",
    "squared_distance",
    "squared_pairwise_distance",
    "euclidean_distance",
    "euclidean_pairwise_distance",
    "dtw_distance",
    "dtw_pairwise_distance",
    "dtw_cost_matrix",
    "dtw_alignment_path",
    "ddtw_distance",
    "ddtw_pairwise_distance",
    "ddtw_alignment_path",
    "ddtw_cost_matrix",
    "distance",
    "distance_factory",
    "pairwise_distance",
    "euclidean_distance",
    "squared_distance",
    "wdtw_distance",
    "wddtw_distance",
    "edr_distance",
    "erp_distance",
    "msm_distance",
    "lcss_distance",
    "twe_distance",
    "wdtw_alignment_path",
    "lcss_alignment_path",
    "msm_alignment_path",
    "erp_alignment_path",
    "edr_alignment_path",
    "distance_alignment_path_factory",
    "distance_alignment_path",
    "twe_alignment_path",
    "wddtw_alignment_path",
]

from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._ddtw import (
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_pairwise_distance,
)
from aeon.distances._distance import (
    distance,
    distance_alignment_path,
    distance_alignment_path_factory,
    distance_factory,
    edr_alignment_path,
    edr_distance,
    erp_alignment_path,
    erp_distance,
    lcss_alignment_path,
    lcss_distance,
    msm_alignment_path,
    msm_distance,
    pairwise_distance,
    twe_alignment_path,
    twe_distance,
    wddtw_alignment_path,
    wddtw_distance,
    wdtw_alignment_path,
    wdtw_distance,
)
from aeon.distances._dtw import (
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_pairwise_distance,
)
from aeon.distances._euclidean import euclidean_distance, euclidean_pairwise_distance
from aeon.distances._squared import squared_distance, squared_pairwise_distance
