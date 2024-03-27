"""Test annotation."""

import pytest

from aeon.testing.utils.data_gen.annotation import make_annotation_problem


def test_make_annotation_problem():
    """Test make annotation."""
    y = make_annotation_problem(n_timepoints=10, make_X=False)
    assert y.shape[0] == 10
    with pytest.raises(ValueError):
        make_annotation_problem(n_timepoints=10, make_X=True, estimator_type="Poisson")
    y, X = make_annotation_problem(n_timepoints=10, make_X=True)
    assert y.shape[0] == 10 and X.shape[0] == 10
