"""Tests for numba utils functions related to statistical operations."""

import numpy as np
import pytest

from aeon.utils.numba.stats import (
    gini,
    gini_gain,
    row_count_above_mean,
    row_count_mean_crossing,
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_ppv,
    row_quantile,
    row_quantile25,
    row_quantile75,
    row_slope,
    row_std,
)


@pytest.mark.parametrize("dtype", ["int16", "int32", "int64", "float32", "float64"])
@pytest.mark.parametrize(
    "func, expected, kwargs",
    [
        (row_mean, [1.5, 2.0], {}),
        (row_std, [np.sqrt(5.25), 0.0], {}),
        (row_quantile, [1.5, 2.0], {"q": 0.5}),
        (row_median, [1.5, 2.0], {}),
        (row_quantile25, [-0.25, 2.0], {}),
        (row_quantile75, [3.25, 2.0], {}),
        (row_numba_min, [-2.0, 2.0], {}),
        (row_numba_max, [5.0, 2.0], {}),
        (row_slope, [1.0, 0.0], {}),
        (row_iqr, [3.5, 0.0], {}),
        (row_ppv, [5 / 8, 1.0], {}),
        (row_count_above_mean, [4.0, 0.0], {}),
        (row_count_mean_crossing, [1.0, 0.0], {}),
    ],
)
def test_row_statistics_dtype_and_values(dtype, func, expected, kwargs):
    """Preserve float32 and promote integer outputs without truncating statistics."""
    X = np.array([[-2, -1, 0, 1, 2, 3, 4, 5], [2] * 8], dtype=dtype)
    result = func(X, **kwargs)
    assert result.dtype == (np.float32 if dtype == "float32" else np.float64)
    np.testing.assert_allclose(
        result, expected, rtol=1e-6 if dtype == "float32" else 1e-12
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("scale", [1e-30, 1e-4, 1e9, 1e30])
def test_row_statistics_precision(dtype, scale):
    """Mean, standard deviation and signed slopes retain stable intermediates."""
    X = (np.arange(1, 17, dtype=dtype) * scale).reshape(2, 8)[:, ::-1]
    X[1] *= -1
    reference = X.astype(np.float64)
    times = np.arange(X.shape[1], dtype=np.float64)
    times -= times.mean()
    slopes = ((reference - reference.mean(axis=1, keepdims=True)) @ times) / (
        times @ times
    )
    for func, expected in [
        (row_mean, reference.mean(axis=1)),
        (row_std, reference.std(axis=1)),
        (row_slope, slopes),
    ]:
        result = func(X)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(
            result, expected, rtol=1e-6 if dtype == "float32" else 1e-12, atol=0
        )


def test_gini():
    """Test the method to calculate gini."""
    # Test case: Pure node (all same class)
    y_pure = np.array([1, 1, 1, 1])
    assert gini(y_pure) == 0.0

    # Test case: Impure node with two classes
    y_impure = np.array([1, 1, 2, 2])
    assert gini(y_impure) == 0.5

    # Test case: More impure node with three classes
    y_more_impure = np.array([1, 1, 2, 3])
    gini_score = 1 - ((2 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2)
    assert gini(y_more_impure) == gini_score

    # Test case: All different classes
    y_all_different = np.array([1, 2, 3, 4])
    gini_score_all_diff = 1 - (
        (1 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2 + (1 / 4) ** 2
    )
    assert gini(y_all_different) == gini_score_all_diff

    # Test case: Empty array
    y_empty = np.array([])
    with pytest.raises(ValueError, match="y is empty"):
        gini(y_empty)


def test_gini_gain():
    """Test the method to calculate gini gain of a node."""
    # Split with mixed children
    y = np.array([1, 1, 2, 2, 4, 4, 2, 2])
    y_subs = [np.array([1, 1, 4, 4]), np.array([2, 2, 2, 2])]
    score_y = 1 - ((2 / 8) ** 2 + (4 / 8) ** 2 + (2 / 8) ** 2)
    score = score_y - ((4 / 8) * 0.5 + (4 / 8) * 0)
    assert gini_gain(y, y_subs) == score

    # Split with pure children
    y = np.array([1, 1, 0, 0])
    y_children = [np.array([1, 1]), np.array([0, 0], dtype=y.dtype)]
    assert gini_gain(y, y_children) == gini(y)

    # Test case: Empty array
    y_empty = np.array([])
    with pytest.raises(ValueError, match="y is empty"):
        gini(y_empty)

    # When labels in y_subs do not sum to the same as y
    y_empty = np.array([1, 1, 0, 0])
    y_children = [
        np.array([1, 1]),
        np.array(
            [
                0,
            ],
            dtype=y.dtype,
        ),
    ]
    with pytest.raises(ValueError, match="labels in y_subs must sum"):
        gini_gain(y_empty, y_children)
