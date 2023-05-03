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
    "wdtw_distance",
    "wdtw_pairwise_distance",
    "wdtw_cost_matrix",
    "wdtw_alignment_path",
    "wddtw_distance",
    "wddtw_pairwise_distance",
    "wddtw_alignment_path",
    "wddtw_cost_matrix",
    "lcss_distance",
    "lcss_pairwise_distance",
    "lcss_alignment_path",
    "lcss_cost_matrix",
    "erp_distance",
    "erp_pairwise_distance",
    "erp_alignment_path",
    "erp_cost_matrix",
    "edr_distance",
    "edr_pairwise_distance",
    "edr_alignment_path",
    "edr_cost_matrix",
    "twe_distance",
    "twe_pairwise_distance",
    "twe_alignment_path",
    "twe_cost_matrix",
    "msm_distance",
    "msm_alignment_path",
    "msm_cost_matrix",
    "msm_pairwise_distance",
    "distance",
    "pairwise_distance",
    "alignment_path",
    "cost_matrix",
    "get_cost_matrix_function",
    "get_distance_function",
    "get_pairwise_distance_function",
    "get_alignment_path_function",
]

from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._ddtw import (
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_pairwise_distance,
)
from aeon.distances._distance import (
    alignment_path,
    cost_matrix,
    distance,
    get_alignment_path_function,
    get_cost_matrix_function,
    get_distance_function,
    get_pairwise_distance_function,
    pairwise_distance,
)
from aeon.distances._dtw import (
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_pairwise_distance,
)
from aeon.distances._edr import (
    edr_alignment_path,
    edr_cost_matrix,
    edr_distance,
    edr_pairwise_distance,
)
from aeon.distances._erp import (
    erp_alignment_path,
    erp_cost_matrix,
    erp_distance,
    erp_pairwise_distance,
)
from aeon.distances._euclidean import euclidean_distance, euclidean_pairwise_distance
from aeon.distances._lcss import (
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_distance,
    lcss_pairwise_distance,
)
from aeon.distances._msm import (
    msm_alignment_path,
    msm_cost_matrix,
    msm_distance,
    msm_pairwise_distance,
)
from aeon.distances._squared import squared_distance, squared_pairwise_distance
from aeon.distances._twe import (
    twe_alignment_path,
    twe_cost_matrix,
    twe_distance,
    twe_pairwise_distance,
)
from aeon.distances._wddtw import (
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_pairwise_distance,
)
from aeon.distances._wdtw import (
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_pairwise_distance,
)
