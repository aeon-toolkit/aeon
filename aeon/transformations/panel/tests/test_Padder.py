# -*- coding: utf-8 -*-
"""Test the Padder transformer."""

import numpy as np

from aeon.transformations.panel.padder import PaddingTransformer


def test_padding_equal_length():
    """Test the dimensions after padding."""
    # Create 3d numpy array (10, 2, 20)
    X = np.random.rand(10, 2, 20)
    padding_transformer = PaddingTransformer()
    X_padded = padding_transformer.fit_transform(X)
    #  Padding should not change shape of equal length
    assert X_padded.shape == X.shape


def test_padding_parameterised_transformer():
    """Test padding to user determined length."""
    # load data
    X = np.random.rand(10, 2, 20)
    padding_transformer = PaddingTransformer(pad_length=120)
    X_padded = padding_transformer.fit_transform(X)
    #  Series now of length 120
    assert X_padded.shape == (X.shape[0], X.shape[1], 120)


def test_padding_fill_value_transformer():
    """Test full fill padding."""
    X = np.random.rand(10, 2, 20)
    padding_transformer = PaddingTransformer(pad_length=120, fill_value=42)
    X_padded = padding_transformer.fit_transform(X)

    assert X_padded.shape == (X.shape[0], X.shape[1], 120)
    assert X_padded[0][0][100] == 42
