"""Tests for numba functions."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from aeon.utils.numba.general import (
    combinations_1d,
    get_all_subsequences,
    get_subsequence,
    get_subsequence_with_mean_std,
    is_prime,
    normalise_subsequences,
    prime_up_to,
    sliding_mean_std_one_series,
    slope_derivative,
    slope_derivative_2d,
    slope_derivative_3d,
    unique_count,
    z_normalise_series,
    z_normalise_series_2d,
    z_normalise_series_2d_with_mean_std,
    z_normalise_series_3d,
    z_normalise_series_with_mean,
    z_normalise_series_with_mean_std,
)

DATATYPES = ["int32", "int64", "float32", "float64"]


@pytest.mark.parametrize("type", DATATYPES)
def test_unique_count(type):
    """Test numba unique count."""
    a = np.array([2, 0, 2, 2, 1, 1, 0, 2, 2, 1], dtype=type)
    unique_expected = [0, 1, 2]
    count_expected = [2, 3, 5]
    a_result = unique_count(a)
    assert_array_equal(a_result[0], unique_expected)
    assert_array_equal(a_result[1], count_expected)


@pytest.mark.parametrize("type", DATATYPES)
def test_z_normalise_series_with_mean_std(type):
    """Test z-normalization of a series using mean and standard deviation."""
    a = np.array([2, 2, 2], dtype=type)
    a_expected = np.array([0, 0, 0], dtype=type)
    a_result = z_normalise_series_with_mean_std(a, a.mean(), a.std())
    assert_array_equal(a_result, a_expected)


@pytest.mark.parametrize("type", DATATYPES)
def test_z_normalise_series(type):
    """Test the function z_normalise_series."""
    a = np.array([2, 2, 2], dtype=type)
    a_expected = np.array([0, 0, 0], dtype=type)
    a_result = z_normalise_series(a)
    assert_array_equal(a_result, a_expected)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_preserves_float_precision(dtype):
    """float32 input stays float32, integer input is promoted to float64."""
    a = np.array([1, 2, 2, 3, 3, 3], dtype=dtype)
    expected = np.float32 if dtype == "float32" else np.float64
    assert z_normalise_series(a).dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_with_mean_preserves_float_precision(dtype):
    """float32 input stays float32, integer input is promoted to float64."""
    a = np.array([1, 2, 2, 3, 3, 3], dtype=dtype)
    expected = np.float32 if dtype == "float32" else np.float64
    assert z_normalise_series_with_mean(a, a.mean()).dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_with_mean_std_preserves_float_precision(dtype):
    """float32 input stays float32, integer input is promoted to float64."""
    a = np.array([1, 2, 2, 3, 3, 3], dtype=dtype)
    expected = np.float32 if dtype == "float32" else np.float64
    assert z_normalise_series_with_mean_std(a, a.mean(), a.std()).dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_constant_input_preserves_float_precision(dtype):
    """The below-threshold std branch must return the same dtype as the main one."""
    a = np.full(6, 2, dtype=dtype)
    expected = np.float32 if dtype == "float32" else np.float64
    assert z_normalise_series(a).dtype == expected
    assert z_normalise_series_with_mean(a, a.mean()).dtype == expected
    assert z_normalise_series_with_mean_std(a, a.mean(), a.std()).dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_2d_preserves_float_precision(dtype):
    """float32 input stays float32, integer input is promoted to float64."""
    X = np.array([[1, 2, 2, 3, 3, 3], [5, 6, 6, 7, 7, 7]], dtype=dtype)
    expected = np.float32 if dtype == "float32" else np.float64
    assert z_normalise_series_2d(X).dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_3d_preserves_float_precision(dtype):
    """float32 input stays float32, integer input is promoted to float64."""
    X = np.array([[[1, 2, 2, 3, 3, 3], [5, 6, 6, 7, 7, 7]]], dtype=dtype)
    expected = np.float32 if dtype == "float32" else np.float64
    assert z_normalise_series_3d(X).dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_2d_with_mean_std_preserves_float_precision(dtype):
    """float32 input stays float32, integer input is promoted to float64."""
    X = np.array([[1, 2, 2, 3, 3, 3], [5, 6, 6, 7, 7, 7]], dtype=dtype)
    mean = X.mean(axis=1).astype(np.float64)
    std = X.std(axis=1).astype(np.float64)
    expected = np.float32 if dtype == "float32" else np.float64
    assert z_normalise_series_2d_with_mean_std(X, mean, std).dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_z_normalise_series_3d_values_unchanged(dtype):
    """Preserving the input precision must not change the normalised values."""
    X = np.array(
        [
            [[1, 2, 2, 3, 3, 3], [5, 6, 6, 7, 7, 7]],
            [[4, 4, 4, 3, 3, 1], [8, 8, 8, 7, 7, 5]],
        ],
        dtype=dtype,
    )
    result = z_normalise_series_3d(X)
    expected = (X - X.mean(axis=-1, keepdims=True)) / X.std(axis=-1, keepdims=True)
    assert_array_almost_equal(result, expected, decimal=5)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_subsequence(dtype):
    """Test the extraction of subsequences from a 1D array."""
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=dtype)
    # get_subsequence(i_start, length, dilation)
    sub = get_subsequence(x, 2, 3, 1)
    assert_array_equal(x[:, [2, 3, 4]], sub)
    sub = get_subsequence(x, 2, 3, 3)
    assert_array_equal(x[:, [2, 5, 8]], sub)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_subsequence_with_mean_std(dtype):
    """Test the extraction of subsequences with mean and std from a 1D array."""
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=dtype)
    # i_start, length, dilation
    sub, mean, std = get_subsequence_with_mean_std(x, 2, 3, 1)
    assert_array_equal(x[:, [2, 3, 4]], sub)
    assert_array_almost_equal(mean, sub.mean(axis=1))
    assert_array_almost_equal(std, sub.std(axis=1))

    sub, mean, std = get_subsequence_with_mean_std(x, 2, 3, 3)
    assert_array_equal(x[:, [2, 5, 8]], sub)
    assert_array_almost_equal(mean, sub.mean(axis=1))
    assert_array_almost_equal(std, sub.std(axis=1))


@pytest.mark.parametrize("dtype", DATATYPES)
def test_sliding_mean_std_one_series(dtype):
    """Test sliding mean and standard deviation computations on a series."""
    X = np.random.rand(3, 150).astype(dtype)
    for length in [5, 50]:
        for dilation in [1, 3]:
            mean, std = sliding_mean_std_one_series(X, length, dilation)
            for i_sub in range(X.shape[1] - (length - 1) * dilation):
                _idx = [i_sub + j * dilation for j in range(length)]
                if dtype == "float32":
                    assert_array_almost_equal(
                        X[:, _idx].mean(axis=1), mean[:, i_sub], decimal=4
                    )
                    assert_array_almost_equal(
                        X[:, _idx].std(axis=1), std[:, i_sub], decimal=4
                    )
                else:
                    assert_array_almost_equal(X[:, _idx].mean(axis=1), mean[:, i_sub])
                    assert_array_almost_equal(X[:, _idx].std(axis=1), std[:, i_sub])

    # Test error on wrong dimension
    error_str = "Invalid input parameter for sliding mean and std computations"
    with pytest.raises(ValueError, match=error_str):
        mean, std = sliding_mean_std_one_series(X, 100, 3)

    with pytest.raises(ValueError, match=error_str):
        mean, std = sliding_mean_std_one_series(X, 100, 3)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_float_output_dtype_follows_input_precision(dtype):
    """Test helpers preserve float precision and promote integer input."""
    X = np.arange(24, dtype=dtype).reshape(2, 12)
    expected_dtype = np.float32 if dtype == "float32" else np.float64

    subsequence = get_subsequence(X, 1, 4, 2)
    subsequence_with_stats = get_subsequence_with_mean_std(X, 1, 4, 2)
    sliding_stats = sliding_mean_std_one_series(X, 4, 2)
    outputs = {
        "get_subsequence": (subsequence,),
        "get_subsequence_with_mean_std": subsequence_with_stats,
        "sliding_mean_std_one_series": sliding_stats,
        "slope_derivative": (slope_derivative(X[0]),),
        "slope_derivative_2d": (slope_derivative_2d(X),),
        "slope_derivative_3d": (slope_derivative_3d(X[np.newaxis]),),
    }

    for function_name, function_outputs in outputs.items():
        for output in function_outputs:
            assert output.dtype == expected_dtype, function_name


@pytest.mark.parametrize("dtype", DATATYPES)
def test_combinations_1d(dtype):
    """Test combinations of elements from two 1D arrays."""
    x = np.array([1, 1, 2, 2, 3, 3, 9, 4, 7, 9, 9], dtype=dtype)
    y = np.array([1, 1, 3, 5, 1, 3, 9, 2, 9, 9, 7], dtype=dtype)
    combs = combinations_1d(x, y)
    true_combs = np.array(
        [[1, 1], [2, 3], [2, 5], [3, 1], [3, 3], [9, 9], [4, 2], [7, 9], [9, 7]],
        dtype=dtype,
    )
    assert_array_equal(combs, true_combs)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalise_subsequences(dtype):
    """Test 3d z-normalization."""
    X = np.asarray([[[1, 1, 1]], [[1, 1, 1]]], dtype=dtype)
    # Transpose as this function expect means and std in (n channels, n_subsequence)
    X_norm = normalise_subsequences(X, X.mean(axis=2).T, X.std(axis=2).T)
    assert np.all(X_norm == 0)
    assert np.all(X.shape == X_norm.shape)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalise_subsequences_preserves_float_precision(dtype):
    """float32 input stays float32, integer input is promoted to float64."""
    X = np.asarray([[[1, 2, 3, 4]], [[4, 5, 6, 8]]], dtype=dtype)
    X_norm = normalise_subsequences(X, X.mean(axis=2).T, X.std(axis=2).T)
    expected = np.float32 if dtype == "float32" else np.float64
    assert X_norm.dtype == expected


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalise_subsequences_values_unchanged(dtype):
    """Preserving the input precision must not change the normalised values."""
    X = np.asarray([[[1, 2, 3, 4]], [[4, 5, 6, 8]]], dtype=dtype)
    X_norm = normalise_subsequences(X, X.mean(axis=2).T, X.std(axis=2).T)
    expected = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)
    assert_array_almost_equal(X_norm, expected, decimal=5)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalise_subsequences_zeroes_below_std_threshold(dtype):
    """Constant subsequences must still fall back to the zero-filled default."""
    X = np.asarray([[[2, 2, 2, 2]], [[1, 2, 3, 4]]], dtype=dtype)
    X_norm = normalise_subsequences(X, X.mean(axis=2).T, X.std(axis=2).T)
    assert np.all(X_norm[0] == 0)
    assert not np.all(X_norm[1] == 0)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_all_subsequences(dtype):
    """Test generation of all subsequences."""
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype)
    length = 3
    dilation = 1
    X_subs = get_all_subsequences(X, length, dilation)
    X_true = np.asarray(
        [
            [[1, 2, 3]],
            [[2, 3, 4]],
            [[3, 4, 5]],
            [[4, 5, 6]],
            [[5, 6, 7]],
            [[6, 7, 8]],
        ],
        dtype=dtype,
    )
    assert_array_equal(X_subs, X_true)

    length = 3
    dilation = 2
    X_subs = get_all_subsequences(X, length, dilation)
    X_true = np.asarray(
        [
            [[1, 3, 5]],
            [[2, 4, 6]],
            [[3, 5, 7]],
            [[4, 6, 8]],
        ],
        dtype=dtype,
    )
    assert_array_equal(X_subs, X_true)


def test_prime_up_to():
    """Test the generation of prime numbers up to a specified limit."""
    true_primes_to_100 = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    primes = prime_up_to(100)
    assert_array_equal(true_primes_to_100, primes)


def test_is_prime():
    """Test the determination of prime numbers."""
    true_primes_to_100 = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    for n in np.arange(100):
        if n in true_primes_to_100:
            assert is_prime(n)
        else:
            assert not is_prime(n)
