"""Rocket test code."""

import numpy as np

from aeon.datasets import load_unit_test
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)

expected_unit_test = {}
expected_unit_test["MiniRocket"] = np.array(
    [
        [0.5833333, 0.7083333, 0.16666667, 0.5833333, 0.8333333],
        [0.5416667, 0.7083333, 0.16666667, 0.5416667, 0.7916667],
        [0.5833333, 0.7083333, 0.20833333, 0.5833333, 0.7916667],
        [0.45833334, 0.6666667, 0.16666667, 0.5, 0.75],
        [0.5416667, 0.7916667, 0.125, 0.5416667, 0.8333333],
    ]
)


def test_datatype_input():
    """Test rocket variants accept all input types."""
    uni_rockets = [
        Rocket(num_kernels=100),
        MiniRocket(num_kernels=100),
        MultiRocket(num_kernels=100),
    ]

    shape = (10, 1, 20)
    types = [np.float32, np.float64, np.int32, np.int64]
    for r in uni_rockets:
        for t in types:
            X = np.random.rand(*shape).astype(t)
            r.fit_transform(X)
            r.fit(X)
            r.transform(X)
    multi_rockets = [
        MiniRocketMultivariate(num_kernels=100),
        MultiRocketMultivariate(num_kernels=100),
    ]
    shape = (10, 3, 20)
    for r in multi_rockets:
        for t in types:
            X = np.random.rand(*shape).astype(t)
            r.fit_transform(X)
            r.fit(X)
            r.transform(X)


def test_expected_unit_test():
    """Test MiniRocket on unit test data."""
    X, _ = load_unit_test(split="train")
    mr = MiniRocket(random_state=0)
    X2 = mr.fit_transform(X)
    np.testing.assert_allclose(X2[:5, :5], expected_unit_test["MiniRocket"], rtol=1e-6)
