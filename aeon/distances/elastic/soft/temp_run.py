from math import gamma

import numpy as np

from aeon.datasets import load_gunpoint
from aeon.distances import msm_distance
from aeon.distances.elastic.soft._soft_dtw import (
    soft_dtw_alignment_matrix,
    soft_dtw_distance,
    soft_dtw_pairwise_distance,
)
from aeon.distances.elastic.soft._soft_dtw_divergence import soft_dtw_divergence_grad_x
from aeon.distances.elastic.soft._soft_msm import (
    soft_msm_alignment_matrix,
    soft_msm_distance,
    soft_msm_grad_x,
    soft_msm_pairwise_distance,
)
from aeon.testing.data_generation import make_example_3d_numpy

if __name__ == "__main__":
    X = make_example_3d_numpy(100, 1, 10, return_y=False, random_state=1)
    # X = np.random.normal(size=(2, 100, 10), scale=100)
    # X, y = load_gunpoint(split="train")
    y = X[1]

    x = X[0]

    # dist_temp = soft_msm_distance(x, y)
    # matrix, dist = soft_msm_alignment_matrix(x, y)
    # print(f"Distance: {dist_temp}")
    # print(f"Matrix: {matrix}")
    #
    # dtw_matrix, dtw_dist = soft_dtw_alignment_matrix(x, y)

    import time

    start = time.time()

    # msm_dist = msm_distance(x, y)
    # soft_msm_distance_1 = soft_msm_distance(x, y)
    # soft_msm_distance_2 = soft_msm_distance(x, y, gamma=0.00001)

    alignment_matrix, dist = soft_dtw_alignment_matrix(x, x, gamma=1.0)
    # div_alig, div_dist = soft_dtw_distance(x, x, gamma=1.0)

    end = time.time()
    print(f"Time: {end - start}")
    stop = ""

    # Time: 24.509012937545776
