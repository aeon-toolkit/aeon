"""Test segmentation data generation."""

import numpy as np
import pytest
from numpy import array_equal

from aeon.testing.data_generation.segmentation import (
    GenBasicGauss,
    label_piecewise_normal,
    labels_with_repeats,
    piecewise_multinomial,
    piecewise_normal,
    piecewise_normal_multivariate,
    piecewise_poisson,
)


def test_segmentation_generation():
    """Test the piecewise generation functions."""
    X = piecewise_normal_multivariate(
        means=[[1, 1], [2, 2], [3, 3]], lengths=[2, 3, 1], random_state=2
    )
    assert isinstance(X, np.ndarray)
    exp = np.array(
        [
            [0.58324215, 0.94373317],
            [-1.1361961, 2.64027081],
            [0.20656441, 1.15825263],
            [2.50288142, 0.75471191],
            [0.94204778, 1.09099239],
            [3.55145404, 5.29220801],
        ]
    )
    assert np.allclose(X, exp)
    X = piecewise_normal([1, 2, 3], lengths=[2, 4, 8], random_state=42)
    exp = np.array(
        [
            1.49671415,
            0.8617357,
            2.64768854,
            3.52302986,
            1.76584663,
            1.76586304,
            4.57921282,
            3.76743473,
            2.53052561,
            3.54256004,
            2.53658231,
            2.53427025,
            3.24196227,
            1.08671976,
        ]
    )
    assert np.allclose(X, exp)

    X = piecewise_normal(
        [1, 2, 3], lengths=[2, 4, 8], std_dev=[0, 0.5, 1.0], random_state=42
    )
    exp = np.array(
        [
            1.0,
            1.0,
            2.32384427,
            2.76151493,
            1.88292331,
            1.88293152,
            4.57921282,
            3.76743473,
            2.53052561,
            3.54256004,
            2.53658231,
            2.53427025,
            3.24196227,
            1.08671976,
        ]
    )
    assert np.allclose(X, exp)
    X = piecewise_multinomial(
        20, lengths=[3, 2], p_vals=[[1 / 4, 3 / 4], [3 / 4, 1 / 4]], random_state=42
    )
    exp = np.array([[4, 16], [8, 12], [6, 14], [15, 5], [17, 3]])
    assert np.allclose(X, exp)
    X = piecewise_multinomial(10, lengths=[2, 4, 8], p_vals=[[1, 0], [0, 1], [1, 0]])
    exp = np.array(
        [
            [10, 0],
            [10, 0],
            [0, 10],
            [0, 10],
            [0, 10],
            [0, 10],
            [10, 0],
            [10, 0],
            [10, 0],
            [10, 0],
            [10, 0],
            [10, 0],
            [10, 0],
            [10, 0],
        ]
    )
    assert np.allclose(X, exp)
    X = piecewise_poisson(lambdas=[1, 2, 3], lengths=[2, 4, 8], random_state=42)
    exp = np.array([1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 2, 4, 2, 1])
    assert np.allclose(X, exp)
    X = piecewise_poisson(lambdas=[1, 3, 6], lengths=[2, 4, 8], random_state=42)
    exp = np.array([1, 2, 1, 3, 3, 2, 5, 5, 6, 4, 4, 9, 3, 5])
    assert np.allclose(X, exp)


def test_label_generation():
    """Test label generation."""
    y = labels_with_repeats(means=[1.0, 2.0, 3.0], std_dev=[0.5, 1.0, 2.0])
    exp = np.array([0, 1, 2])
    assert np.allclose(y, exp)
    y = label_piecewise_normal([1, 2, 3], lengths=[10, 10, 10], std_dev=[0.5, 1.0, 2.0])
    exp = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
    )
    assert np.allclose(y, exp)
    gen = GenBasicGauss([1, 2, 3], lengths=[2, 4, 8], random_state=42)
    X = gen.sample()
    exp = np.array(
        [
            1.49671415,
            0.8617357,
            2.64768854,
            3.52302986,
            1.76584663,
            1.76586304,
            4.57921282,
            3.76743473,
            2.53052561,
            3.54256004,
            2.53658231,
            2.53427025,
            3.24196227,
            1.08671976,
        ]
    )
    assert np.allclose(X, exp)


@pytest.mark.parametrize(
    "lambdas, lengths, random_state, output",
    [
        ([1, 2, 3], [2, 4, 8], 42, [1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 2, 4, 2, 1]),
        ([1, 3, 6], [2, 4, 8], 42, [1, 2, 1, 3, 3, 2, 5, 5, 6, 4, 4, 9, 3, 5]),
    ],
)
def test_piecewise_poisson(lambdas, lengths, random_state, output):
    """Test piecewise_poisson fuction returns the expected Poisson distributed array."""
    assert array_equal(piecewise_poisson(lambdas, lengths, random_state), output)
