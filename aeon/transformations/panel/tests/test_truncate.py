# -*- coding: utf-8 -*-
"""Test Truncator transformer."""

import numpy as np
import pytest

from aeon.datasets import (
    load_basic_motions,
    load_japanese_vowels,
    load_plaid,
    load_unit_test,
)
from aeon.transformations.panel.truncate import TruncationTransformer


@pytest.mark.parametrize(
    "loader", [load_japanese_vowels, load_plaid, load_unit_test, load_basic_motions]
)
def test_truncation_transformer(loader):
    """Test truncation to the shortest series length."""
    X_train, y_train = loader(split="train")
    n_cases = len(X_train)
    n_channels = len(X_train[0])
    truncated_transformer = TruncationTransformer(5)
    Xt = truncated_transformer.fit_transform(X_train)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_channels, 5)


def test_truncation_transformer_shortest():
    """Test truncation to the shortest series length."""
    # load data
    X_train, y_train = load_japanese_vowels(split="train")
    n_cases = len(X_train)
    n_channels = len(X_train[0])
    truncated_transformer = TruncationTransformer()
    Xt = truncated_transformer.fit_transform(X_train)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_channels, 7)
