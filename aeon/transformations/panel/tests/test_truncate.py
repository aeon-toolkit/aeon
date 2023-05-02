# -*- coding: utf-8 -*-
"""Test Truncator transformer."""

import numpy as np

from aeon.datasets import load_plaid
from aeon.transformations.panel.truncate import TruncationTransformer


def test_truncation_transformer():
    """Test truncation to the shortest series length."""
    # load data
    X_train, y_train = load_plaid(split="train")
    n_cases = len(X_train)
    n_channels = len(X_train[0])
    truncated_transformer = TruncationTransformer(5)
    Xt = truncated_transformer.fit_transform(X_train)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_channels, 5)
