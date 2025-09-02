from aeon.datasets import load_gunpoint
from aeon.distances.elastic.soft._soft_msm import (
    soft_msm_alignment_matrix,
    soft_msm_distance,
)
from aeon.testing.data_generation import make_example_3d_numpy

if __name__ == "__main__":
    X = make_example_3d_numpy(10, 1, 10, return_y=False, random_state=1)
    X, y = load_gunpoint(split="train")

    x = X[0]
    y = X[1]

    dist_temp = soft_msm_distance(x, y)
    matrix, dist = soft_msm_alignment_matrix(x, y)
    print(f"Distance: {dist_temp}")
    print(f"Matrix: {matrix}")
    stop = ""
