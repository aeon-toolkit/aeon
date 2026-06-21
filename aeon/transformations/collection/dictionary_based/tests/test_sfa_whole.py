"""Test for SFAWhole transformations on time series."""

import numpy as np

from aeon.transformations.collection.dictionary_based import SFAWhole


def test_sfa_whole_fit_transform():
    """Test SFAWhole fit_transform produces SFA words."""
    X = np.random.RandomState(0).normal(size=(10, 1, 32))
    y = np.random.RandomState(0).randint(0, 2, 10)

    sfa = SFAWhole(word_length=4, alphabet_size=4)
    words = sfa.fit_transform(X, y)

    assert words is not None


def test_sfa_whole_transform_after_fit():
    """Test SFAWhole transform after a separate fit call."""
    X = np.random.RandomState(0).normal(size=(10, 1, 32))
    y = np.random.RandomState(0).randint(0, 2, 10)

    sfa = SFAWhole(word_length=4, alphabet_size=4)
    sfa.fit(X, y)
    words = sfa.transform(X)

    assert words is not None


def test_sfa_whole_get_test_params():
    """Test the default test parameters are valid and usable."""
    params = SFAWhole._get_test_params()
    sfa = SFAWhole(**params)
    assert isinstance(sfa, SFAWhole)
