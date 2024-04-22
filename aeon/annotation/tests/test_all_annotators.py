"""Tests for aeon annotators."""

__maintainer__ = []
__all__ = []

import numpy as np
import pandas as pd
import pytest

from aeon.registry import all_estimators
from aeon.testing.utils.data_gen import make_annotation_problem
from aeon.utils.validation._dependencies import _check_estimator_deps

ALL_ANNOTATORS = all_estimators(estimator_types="series-annotator", return_names=False)


@pytest.mark.parametrize("Estimator", ALL_ANNOTATORS)
def test_output_type(Estimator):
    """Test annotator output type."""
    if not _check_estimator_deps(Estimator, severity="none"):
        return None

    estimator = Estimator.create_test_instance()

    arg = make_annotation_problem(
        n_timepoints=50, estimator_type=estimator.get_tag("distribution_type")
    )
    estimator.fit(arg)
    arg = make_annotation_problem(
        n_timepoints=10, estimator_type=estimator.get_tag("distribution_type")
    )
    y_pred = estimator.predict(arg)
    assert isinstance(y_pred, (pd.Series, np.ndarray))
