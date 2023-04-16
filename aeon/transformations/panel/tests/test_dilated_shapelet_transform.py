# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:59:04 2023

@author: antoi
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.transformations.panel.dilated_shapelet_transform import (
    compute_normalized_shapelet_dist_vector,
    compute_shapelet_dist_vector,
)

DATATYPES = ("int32", "int64", "float32", "float64")


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_normalized_shapelet_dist_vector(dtype):
    # Constant case is tested with dtype int
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            X = (np.random.rand(3, 50)).astype(dtype)
            values = np.random.rand(3, length).astype(dtype)
            d_vect = compute_normalized_shapelet_dist_vector(
                X, values, length, dilation, values.mean(axis=1), values.std(axis=1)
            )
            norm_values = values - values.mean(axis=1, keepdims=True)
            for i_channel in range(X.shape[0]):
                if values[i_channel].std() > 0:
                    norm_values[i_channel] /= values[i_channel].std()
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                for i_channel in range(X.shape[0]):
                    norm_sub = X[i_channel, _idx]
                    norm_sub = norm_sub - norm_sub.mean()
                    if norm_sub.std() > 0:
                        norm_sub /= norm_sub.std()
                    true_vect[i_sub] += ((norm_values[i_channel] - norm_sub) ** 2).sum()
            if dtype == "float32":
                # Fastmath with float32 can sometime produce different of approx 0.0005
                # Any way to compensate this ?
                assert_array_almost_equal(d_vect, true_vect, decimal=3)
            else:
                assert_array_almost_equal(d_vect, true_vect)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_dist_vector(dtype):
    X = np.random.rand(3, 50).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            values = np.random.rand(3, length).astype(dtype)
            d_vect = compute_shapelet_dist_vector(X, values, length, dilation)
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                for i_channel in range(X.shape[0]):
                    _sub = X[i_channel, _idx]
                    true_vect[i_sub] += ((values[i_channel] - _sub) ** 2).sum()
            assert_array_almost_equal(d_vect, true_vect)
