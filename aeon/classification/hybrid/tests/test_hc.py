# -*- coding: utf-8 -*-
"""Tests for HC1."""
import pytest

from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from aeon.tests._config import PR_TESTING
from aeon.utils._testing.collection import make_2d_test_data


@pytest.mark.skipif(PR_TESTING, reason="slow test, run overnight only")
def test_hc1():
    """Test HC1."""
    X, y = make_2d_test_data(n_cases=20, n_timepoints=10, n_labels=2)
    hc1 = HIVECOTEV1(stc_params={"transform_limit_in_minutes": 0.1}, verbose=True)
    hc1.fit(X, y)
    assert hc1._tsf_params == {"n_estimators": 500}
    assert hc1._rise_params == {"n_estimators": 500}
    assert hc1._cboss_params == {}


@pytest.mark.skipif(PR_TESTING, reason="slow test, run overnight only")
def test_hc2():
    """Test HC2."""
    X, y = make_2d_test_data(n_cases=20, n_timepoints=10, n_labels=2)
    hc2 = HIVECOTEV2(stc_params={"transform_limit_in_minutes": 0.1}, verbose=True)
    hc2.fit(X, y)
    assert hc2._drcif_params == {"n_estimators": 500}
    assert hc2._arsenal_params == {}
    assert hc2._tde_params == {}
