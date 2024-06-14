"""Rocket test code."""

import numpy as np

from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
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
