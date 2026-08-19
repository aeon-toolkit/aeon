import numpy as np

TEST_DISTANCE_WITH_PARAMS = [
    ("dtw", {"window": 0.2}),
    ("wdtw", {"g": 2.0}),
    ("edr", {"epsilon": 0.8}),
    ("twe", {"nu": 0.2, "lmbda": 0.1}),
    ("msm", {"c": 2.0}),
    ("shape_dtw", {"reach": 4}),
    ("adtw", {"warp_penalty": 0.2}),
    ("soft_dtw", {"gamma": 0.1}),
    ("euclidean", {}),
    ("squared", {}),
    ("manhattan", {}),
    ("minkowski", {"p": 3.0}),
    ("dtw_gi", {"window": 0.2}),
    ("ddtw", {"window": 0.2}),
    ("wddtw", {"g": 1}),
    ("lcss", {"epsilon": 0.5}),
    ("erp", {"g": 0.8}),
    ("sbd", {"standardize": False}),
    ("shift_scale", {"max_shift": 2}),
]

# All the distances that return a full alignment path.
TEST_DISTANCES_WITH_FULL_ALIGNMENT_PATH = [
    (name, params)
    for name, params in TEST_DISTANCE_WITH_PARAMS
    if name in ["dtw", "wdtw", "edr", "twe", "msm", "shape_dtw", "adtw", "soft_dtw"]
]


def _custom_distance_measure(x, y, custom_param=1):
    return np.sum(np.abs(x - y)) + custom_param


TEST_DISTANCE_WITH_CUSTOM_DISTANCE = TEST_DISTANCE_WITH_PARAMS + [
    (_custom_distance_measure, {"custom_param": 10}),
]
